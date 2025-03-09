def compute_feature(batch):
    batch["feature"] = [img.mean() for img in batch["image"]]
    return batch
