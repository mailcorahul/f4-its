import os
import json

import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix, save_npz
import pandas as pd
import sys

import open_clip

def compute_map(predictions, ground_truths, k=None):
    """
    Computes mean Average Precision (mAP) for multilabel retrieval.

    Args:
        predictions (List[List[str]]): List of predicted ranked items for each image.
        ground_truths (List[Set[str]]): List of sets of GT items per image.
        k (int or None): Cutoff for top-K predictions. If None, use full ranked list.

    Returns:
        mean_ap (float): Mean Average Precision over all samples.
        per_sample_ap (List[float]): AP per sample.
    """
    assert len(predictions) == len(ground_truths), "Mismatched lengths."

    all_ap = []

    for pred, gt in zip(predictions, ground_truths):
        if len(gt) == 0:
            continue  # skip samples with no GT

        if k is not None:
            pred = pred[:k]

        num_relevant = 0
        precisions = []
        # print(pred, gt)
        for i, item in enumerate(pred):
            # print("[/] item", item)
            if (item in gt) or (item + "s" in gt):
                num_relevant += 1
                precision_at_i = num_relevant / (i + 1)
                # print("[/] precision_at_i", precision_at_i)
                precisions.append(precision_at_i)

        if len(precisions) == 0:
            ap = 0.0
        else:
            ap = sum(precisions) / len(gt)  # divide by number of GT items

        all_ap.append(ap)

    mean_ap = sum(all_ap) / len(all_ap) if all_ap else 0.0
    return mean_ap, all_ap


def load_model(model_name, dataset_name):

    model_dir = "/tmp/models"

    print(f"[/] instantiating {model_name}, {dataset_name}")
    model, _, preprocess = open_clip.create_model_and_transforms(
                    model_name,
                    pretrained=dataset_name,
                    cache_dir=model_dir
            )
    model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    print("[/] clip loaded.")

    return model, preprocess, tokenizer


if __name__ == "__main__":
    device = "cuda"

    image_dir = sys.argv[1]
    gt_sparse_items_file = sys.argv[2]
    image2sparse_gt_file = sys.argv[3]
    image2sparsepred_file = sys.argv[4]
    should_rerank = sys.argv[5]
    if should_rerank == "True":
        should_rerank = True
    else:
        should_rerank = False

    model_name = sys.argv[6]
    dataset_name = sys.argv[7]

    model, preprocess, tokenizer = load_model(model_name, dataset_name)

    image_batch_size = 1
    caption_batch_size = 128

    # === LOAD CAPTIONS ===
    with open(gt_sparse_items_file) as f:
        all_captions = json.load(f)
    num_captions = len(all_captions)
    print(f"Loaded {num_captions} captions.")

    with open(image2sparsepred_file) as f:
        image2sparse_pred = json.load(f)

    with open(image2sparse_gt_file) as f:
        image2sparse_gt = json.load(f)

    image_filenames, all_pred_captions = [], []
    dense_captions = []
    word_counts = []
    for image, caption in image2sparse_pred.items():
        image_filenames.append(image)
        all_pred_captions.append(caption)
        words = caption.split(" ")
        word_counts.append(len(words))

    # print("[/] word distribution")
    # print("mean: ", np.mean(word_counts))
    # print("median", np.median(word_counts))
    # print("min", np.min(word_counts))
    # print("max:", np.max(word_counts))

    image_paths = [os.path.join(image_dir, fname) for fname in image_filenames]
    num_images = len(image_paths)
    print(f"✅ Loaded {num_images} image paths.")


    row_indices = []
    col_indices = []
    all_max_indices = []

    print("[/] extracting text features...")
    all_text_features = []
    caption_features = []
    for cap_start in tqdm(range(0, num_captions, caption_batch_size), desc="text batches"):
        cap_end = min(cap_start + caption_batch_size, num_captions)
        caption_batch = tokenizer(all_captions[cap_start:cap_end]).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = model.encode_text(caption_batch)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            all_text_features.append(text_features)
            caption_features.extend(text_features)

    # all_text_features = torch.stack(all_text_features)
    print("[/] text features shape", len(all_text_features))
    print("[/] flattened caption features shape", len(caption_features))

    csv_data = {}

    gt_set, prediction_set = [], []
    skipped = 0

    for image_start in tqdm(range(0, num_images, image_batch_size), desc="Image Batches"):
        image_end = min(image_start + image_batch_size, num_images)
        batch_paths = image_paths[image_start:image_end]

        # Load images
        image_batch = []
        valid_indices = []
        gemini_batch = []
        gemini_full_batch = []
        for idx, path in enumerate(batch_paths):
            try:
                img = preprocess(Image.open(path))
                image_batch.append(img)
                valid_indices.append(idx)

                img_name = os.path.split(path)[-1]
                gemini_text = image2sparse_pred[img_name]
                gt_sparse_caption = image2sparse_gt[img_name]

                text_splits = []
                if "," in gemini_text:
                    text_splits = gemini_text.split(",")
                else:
                    text_splits = [gemini_text]

                gemini_batch.append(text_splits)
                gemini_full_batch.append(gemini_text)

            except Exception as e:
                print(f"Failed to load image {path}: {e}")
                break

        #print("[/] images loaded.")
        if not image_batch or len(image_batch) == 0 or len(gemini_batch) == 0:
            skipped += 1
            continue

        # print("[/] gemini batch", gemini_batch)
        # print("[/] batch paths", batch_paths)

        w_img = 0.7
        w_text = 0.3

        batch_images = torch.stack(image_batch).to(device)
        gemini_tokens = tokenizer(gemini_full_batch).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(batch_images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            gemini_features = model.encode_text(gemini_tokens)
            gemini_features /= gemini_features.norm(dim=-1, keepdim=True)

            fused_features = w_img * image_features + w_text * gemini_features
            # fused_features /= fused_features.norm(dim=-1, keepdim=True)

        gemini_text_features = []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for gemini_text in gemini_batch[0]:
                # print("[/] embedding", gemini_text)
                gemini_tokens = tokenizer(gemini_text.strip()).to(device)
                gemini_features = model.encode_text(gemini_tokens)
                gemini_features /= gemini_features.norm(dim=-1, keepdim=True)
                gemini_text_features.append(gemini_features)


        c_img = 0.2
        c_text = 0.8
        if not should_rerank:
            all_logits = []
            for text_features in all_text_features:
                # latest diff
                fused_caption_features = c_text * text_features + c_img * image_features
                # ends
                logits = (100.0 * fused_features @ fused_caption_features.T).cpu()
                # print("Label logits:", logits.size())
                # max_indices = torch.argmax(text_probs, dim=1)
                # print(max_indices)
                # print(max_indices.size())
                all_logits.append(logits)

            all_logits = torch.cat(all_logits, dim=1)
            probs = all_logits.softmax(dim=-1)
            # print("[/] probs", probs)
            # print("probs size", probs.size())

            # Top-k selection
            k = len(gemini_batch[0])
            topk_indices = torch.topk(probs, k=k, dim=1).indices
            rerank_probs = []
            # print("topk_indices", topk_indices)
            # print(topk_indices.size())
            topk_indices_list = topk_indices[0].tolist()

        elif should_rerank:
            k = 10
            rerank_probs = []
            gemini_found = []
            final_indices = []

            g_img, g_text = 0.7, 0.3
            ## FUSE IMAGE WITH INDIVIDUAL FOOD ITEM, PICK top-2 and then rerank using that food item text embedding
            for gidx, gemini_feature in enumerate(gemini_text_features):

                # print("\n[/] ranking for", gemini_batch[0][gidx])
                # loop 1: for every gemini food item, fuse it with image and pick top-k
                all_logits = []
                single_fused_features = g_text * gemini_feature + g_img * image_features
                for text_features in all_text_features:
                    # latest diff
                    fused_caption_features = c_text * text_features + c_img * image_features
                    # ends
                    logits = (100.0 * single_fused_features @ fused_caption_features.T).cpu()
                    all_logits.append(logits)

                all_logits = torch.cat(all_logits, dim=1)
                probs = all_logits
                # print("[/] all probs size", probs.size())

                # pick top-2 captions matching with this gemini text
                k = 2
                topk_indices = torch.topk(probs, k=k, dim=1).indices
                # print("topk_indices", topk_indices)
                # print(topk_indices.size())
                topk_indices_list = topk_indices[0].tolist()

                max_prob = -1
                gi = -1
                # loop 2: for the top-2 captions, rerank them and pick the best caption
                for idx in topk_indices_list:
                    logit = (100.0 * gemini_feature @ caption_features[idx].T).cpu()
                    prob = logit[0].item()
                    if prob > max_prob:# and idx not in gemini_found:
                        max_prob = prob
                        gi = idx

                if max_prob != -1 and max_prob > 60 and gi not in gemini_found:
                    # print("[/] max prob found - ", max_prob, gi)
                    rerank_probs.append(max_prob)
                    gemini_found.append(gi)
                    final_indices.append(gi)

            # TODO: pick 1 each for every food item. if this doesn't cross 5, try to pick top-2 and 3 for each of the items post the best picks. top-5 evaluation.

            rerank_len = len(rerank_probs)
            rerank_probs = torch.tensor(rerank_probs).unsqueeze(0)
            # print("rerank probs", rerank_probs)
            # print("rerank idxs", final_indices)
            new_topk_indices = torch.topk(rerank_probs, k=rerank_len, dim=1).indices
            # print("new topk", new_topk_indices)

            topk_indices_reranked = []
            for idx in new_topk_indices[0].tolist():
                topk_indices_reranked.append(final_indices[idx])

            topk_row = ""
            for idx in topk_indices_reranked:
                topk_row += str(idx) + "-"
            topk_row = topk_row[:-1]
            csv_data[np.int64(image_start)] = topk_row

            # print("rerank topk", topk_indices_reranked)
            topk_indices_list = topk_indices_reranked

        prediction = []
        for idx in topk_indices_list:
            prediction.append(all_captions[idx].strip().lower())

        ground_truth = []
        if "," in gt_sparse_caption:
            ground_truth = gt_sparse_caption.split(",")
            for i in range(len(ground_truth)):
                ground_truth[i] = ground_truth[i].strip().lower()
        else:
            ground_truth = [gt_sparse_caption]

        # print(ground_truth, prediction)
        gt_set.append(set(ground_truth))
        prediction_set.append(prediction)

        # print("\n")
        # break


    mean_ap, all_ap = compute_map(prediction_set, gt_set, k=10)
    print("[/] mAP", mean_ap)


    print("[/] number of images skipped", skipped)