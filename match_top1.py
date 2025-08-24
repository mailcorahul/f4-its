import os
import json

import sys
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix, save_npz

import open_clip


def load_model(model_name, dataset_name):

    model_dir = "/tmp/models/"

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

    image_dir = sys.argv[1]#"raw_images_synth"#"raw_images"#
    gt_dense_caption_file = sys.argv[2]#"filename_to_caption_synth.json" # "filename_to_caption_english.json"#
    gt_sparse_caption_file = sys.argv[3]
    pred_dense_caption_file = sys.argv[4]
    pred_sparse_caption_file = sys.argv[5]
    is_sparse = sys.argv[6]
    is_pred_sparse = sys.argv[7]
    should_fuse = sys.argv[8]
    model_name = sys.argv[9]
    dataset_name = sys.argv[10]
    w_img = float(sys.argv[11])
    w_text = float(sys.argv[12])

    device = "cuda"
    model, preprocess, tokenizer = load_model(model_name, dataset_name)

    if should_fuse == "True":
        should_fuse = True
    else:
        should_fuse = False

    if is_sparse == "True":
        is_sparse = True
    else:
        is_sparse = False

    if is_pred_sparse == "True":
        is_pred_sparse = True
    else:
        is_pred_sparse = False

    image_batch_size = 128
    caption_batch_size = 128

    # === LOAD CAPTIONS ===
    with open(gt_dense_caption_file, "r") as f:
        img2caption = json.load(f)

    with open(gt_sparse_caption_file) as f:
        img2sparse_gt = json.load(f)

    with open(pred_dense_caption_file) as f:
        image2captions = json.load(f)

    with open(pred_sparse_caption_file) as f:
        image2sparse_caption = json.load(f)

    if is_pred_sparse:
        print("[/] using sparse pred captions...")
        image2captions = image2sparse_caption
        print(list(image2captions.values())[:10])

    caption2img = {}
    for img, caption in img2caption.items():
        img2caption[img] = caption.lower()
        caption2img[caption] = img

    image_filenames, all_captions = [], []
    dense_captions = []
    word_counts = []
    for image, caption in img2caption.items():
        if caption not in all_captions and image in image2captions and image in img2sparse_gt:
            if is_sparse:
                caption = img2sparse_gt[image]
            image_filenames.append(image)
            all_captions.append(caption)
            words = caption.split(" ")
            word_counts.append(len(words))

    print("[/] word distribution")
    # print("mean: ", np.mean(word_counts))
    # print("median", np.median(word_counts))
    # print("min", np.min(word_counts))
    # print("max:", np.max(word_counts))

    gt_preds = []
    for i1, c1, c2 in zip(image_filenames, all_captions, dense_captions):
        gt_preds.append([i1, c1, c2])

    num_captions = len(all_captions)
    print(f"✅ Loaded {num_captions} captions.")

    # === LOAD IMAGE PATHS ===
    image_paths = [os.path.join(image_dir, fname) for fname in image_filenames]
    num_images = len(image_paths)
    print(f"✅ Loaded {num_images} image paths.")

    print(image_filenames[:10])
    print(all_captions[:10])

    row_indices = []
    col_indices = []
    all_max_indices = []

    print("[/] extracting text features...")
    all_text_features = []
    for cap_start in tqdm(range(0, num_captions, caption_batch_size), desc="text batches"):
        cap_end = min(cap_start + caption_batch_size, num_captions)
        caption_batch = tokenizer(all_captions[cap_start:cap_end]).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = model.encode_text(caption_batch)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            all_text_features.append(text_features)

    # all_text_features = torch.stack(all_text_features)
    print("[/] text features shape", len(all_text_features))

    all_topk_indices = []
    all_valid_images = []
    k = 5
    for image_start in tqdm(range(0, num_images, image_batch_size), desc="Image Batches"):
        image_end = min(image_start + image_batch_size, num_images)
        batch_paths = image_paths[image_start:image_end]

        # Load images
        image_batch = []
        valid_images = []
        gemini_batch = []
        for idx, path in enumerate(batch_paths):
            try:
                img = preprocess(Image.open(path))
                img_name = os.path.split(path)[-1]

                # gemini_text = "a"
                if img_name in image2captions:
                    gemini_text = image2captions[img_name]
                    # if len(gemini_text) == "":
                    #     gemini_text = "a"

                image_batch.append(img)
                gemini_batch.append(gemini_text)
                valid_images.append(img_name)

            except Exception as e:
                print(f"⚠️ Failed to load image {path}: {e}")
                continue

        #print("[/] images loaded.")
        if not image_batch:
            continue

        # w_img = 0.7
        # w_text = 0.3

        # print(gemini_batch[:5])
        gemini_tokens = tokenizer(gemini_batch).to(device)
        batch_images = torch.stack(image_batch).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(batch_images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            gemini_features = model.encode_text(gemini_tokens)
            gemini_features /= gemini_features.norm(dim=-1, keepdim=True)

            fused_features = w_img * image_features + w_text * gemini_features
            # fused_features = image_features + gemini_features
            # fused_features /= fused_features.norm(dim=-1, keepdim=True)

        all_logits = []
        c_img = 0.7
        c_text = 0.3
        for text_features in all_text_features:
            # to fuse caption features with a little bit of image features
            # print(text_features.size(), image_features.size())
            # fused_caption_features = c_text * text_features + c_img * image_features

            if should_fuse:
                logits = (100.0 * fused_features @ text_features.T).cpu()
                # logits = (100.0 * fused_features @ fused_caption_features.T).cpu()
            else:
                logits = (100.0 * image_features @ text_features.T).cpu()

            # logits = (100.0 * fused_features @ fused_caption_features.T).cpu()
            # print("Label logits:", logits.size())
            # max_indices = torch.argmax(text_probs, dim=1)
            # print(max_indices)
            # print(max_indices.size())
            all_logits.append(logits)

        all_logits = torch.cat(all_logits, dim=1)
        all_probs = all_logits.softmax(dim=-1)
        # print("probs size", all_probs.size())
        # print("[/] probs", all_probs)
        max_indices = torch.argmax(all_probs, dim=1).tolist()
        max_probs = torch.max(all_probs, dim=1).values.tolist()
        # print(max_probs)

        # ADAPTIVE FUSION IF CLIP's CONFIDENCE IS LOW.
        if False:#max_probs[0] <= 0.5:
            all_logits = []
            for text_features in all_text_features:
                logits = (100.0 * fused_features @ text_features.T).cpu()
                all_logits.append(logits)

            all_logits = torch.cat(all_logits, dim=1)
            all_probs = all_logits.softmax(dim=-1)
            # print("probs size", all_probs.size())
            # print("[/] probs", all_probs)
            fused_max_probs = torch.max(all_probs, dim=1).values.tolist()
            max_indices = torch.argmax(all_probs, dim=1).tolist()
            # if fused_max_probs[0] > max_probs[0]:
            #     max_indices = torch.argmax(all_probs, dim=1).tolist()
                # print("[/] after fusion", fused_max_probs[0])

        if True:#not rerank:
            topk_indices = torch.topk(all_probs, k=k, dim=1).indices
            topk_indices_list = topk_indices.tolist()
        else:
            k = 10
            topk_indices = torch.topk(all_probs, k=k, dim=1).indices
            topk_indices_list = topk_indices.tolist()

        all_topk_indices.extend(topk_indices_list)
        all_max_indices.extend(max_indices)
        all_valid_images.extend(valid_images)

    matches = 0
    for max_idx, img_name in zip(all_max_indices, all_valid_images):
        # gt_caption = dense2sparse[img2caption[img_name]]
        if img_name not in img2sparse_gt: continue

        gt_caption = img2caption[img_name]
        if is_sparse:
            gt_caption = img2sparse_gt[img_name]
        pred_caption = all_captions[max_idx]
        if gt_caption.lower() == pred_caption.lower():
            matches += 1

    num_samples = len(all_max_indices)
    acc = matches / num_samples
    print("[/] top-1 accuracy:")
    print(f"[/] num_samples: {num_samples}, actual matches: {matches}")
    print("[/] accuracy: ", acc)

    matches = 0
    for topk_indices, img_name in zip(all_topk_indices, all_valid_images):
        # gt_caption = dense2sparse[img2caption[img_name]]
        if img_name not in img2sparse_gt: continue

        gt_caption = img2caption[img_name]
        if is_sparse:
            gt_caption = img2sparse_gt[img_name]

        for idx in topk_indices:
            pred_caption = all_captions[idx]
            if gt_caption.lower() == pred_caption.lower():
                matches += 1
                break


    num_samples = len(all_topk_indices)
    acc = matches / num_samples
    print(f"\n[/] top-{k} accuracy:")
    print(f"[/] num_samples: {num_samples}, actual matches: {matches}")
    print("[/] accuracy: ", acc)



