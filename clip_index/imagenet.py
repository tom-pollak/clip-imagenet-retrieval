import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal

from clip_index.text.query import AnnoyImage


class ImagenetDETDataset:
    def __init__(self, imagenet_root: Path, imagenet_synset: Path):
        self.images_root = imagenet_root / "Data/DET"
        self.annotation_root = imagenet_root / "Annotations/DET"

        # Parse Imagenet classes
        with open(imagenet_synset, "r") as f:
            wordlist = f.read().splitlines()

        wordlist = map(lambda l: l.split(" "), wordlist)
        self._synset2desc = dict([(l[0], l[2].replace("_", " ")) for l in wordlist])

    @staticmethod
    def parse_xml_class(xml_path) -> set[str]:
        xml_root = ET.parse(xml_path).getroot()
        classes = set()
        for obj in xml_root.findall("object"):
            cls = obj.find("name")
            assert isinstance(cls, ET.Element)
            classes.add(cls.text)
        return classes

    def synset2desc(self, clss: list[str]) -> list[str]:
        return list(map(lambda cls: self._synset2desc[cls], clss))

    def total_each_class(
        self, split: Literal["train", "val", "test"]
    ) -> dict[str, int]:
        class_count = {}
        xml_dir = self.annotation_root / split
        xml_files = os.listdir(xml_dir)
        for f in xml_files:
            clss = ImagenetDETDataset.parse_xml_class(xml_dir / f)
            nat_clss = self.synset2desc(list(clss))
            for cls in nat_clss:
                if cls in class_count:
                    class_count[cls] += 1
                else:
                    class_count[cls] = 1
        return class_count

    def get_num_images(self, split: Literal["train", "val", "test"]):
        return len(os.listdir(self.images_root / split))

    def get_classes_from_image_path(
        self, image_path: Path, class_type: Literal["wordnet", "desc"] = "desc"
    ) -> list[str]:
        rel_path = image_path.relative_to(self.images_root)
        rel_path_no_ext = os.path.splitext(rel_path)[0]
        wordnet_classes = list(
            self.parse_xml_class(self.annotation_root / (rel_path_no_ext + ".xml"))
        )
        if class_type == "wordnet":
            return wordnet_classes
        else:
            return self.synset2desc(wordnet_classes)

    def add_imagenet_classes_image(self, ann_img: AnnoyImage):
        assert ann_img.image_path is not None
        clss = self.get_classes_from_image_path(
            Path(ann_img.image_path), class_type="desc"
        )
        ann_img.imagenet_classes = set(clss)

    def add_imagenet_classes_queries(self, annoy_queries):
        for images in annoy_queries.values():
            for im in images:
                self.add_imagenet_classes_image(im)
