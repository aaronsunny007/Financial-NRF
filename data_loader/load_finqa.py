import json
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent


def load_finqa(split="test"):

    path = DATA_DIR / f"{split}.json"

    if not path.exists():
        raise FileNotFoundError(
            f"FinQA {split}.json not found at {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    return data


def load_dataset(split="test"):
    """Dataset selected by NRF_DATASET: "finqa" (default) or "tatqa"."""
    import os

    name = os.environ.get("NRF_DATASET", "finqa").lower()
    if name == "finqa":
        return load_finqa(split)
    if name == "tatqa":
        from data_loader.load_tatqa import load_tatqa

        return load_tatqa(split)
    raise ValueError(f"NRF_DATASET must be 'finqa' or 'tatqa', got {name!r}")


if __name__ == "__main__":

    print("=" * 60)
    print("FinQA DATASET CHECK")
    print("=" * 60)

    for split in ["train", "dev", "test"]:

        data = load_finqa(split)

        print(
            f"{split}: {len(data)} examples"
        )

    test = load_finqa("test")

    example = test[0]

    print("\nExample ID:")
    print(example["id"])

    print("\nQuestion:")
    print(example["qa"]["question"])

    print("\nGold Answer:")
    print(example["qa"]["answer"])

    print("\nGold Program:")
    print(example["qa"]["program"])

    print("\nGold Evidence:")
    print(example["qa"]["gold_inds"])

    print("\nDataset loaded successfully!")