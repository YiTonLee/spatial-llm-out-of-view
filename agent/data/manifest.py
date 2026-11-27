"""Where every public dataset comes from, its licence, and which experiment needs it.

Plain: a shopping list. Each entry says where to get one dataset, how big it is, what the licence allows, and which
experiment uses it. `download.py` reads this list; `checksums.json` (next to this file) holds each file's size and
fingerprint (sha256) as we downloaded it on 2026-11-25, so a teammate can confirm their copy is identical.

Technical: DATASETS maps a folder name under <data>/ (default data/public/) to a dict with
  kind      'url' (plain HTTPS, resumable with Range), 'gdrive' (public Google Drive file ids), 'hf' (Hugging Face
            snapshot_download with allow_patterns), 'hf_model', or 'manual' (registration or browser-only download)
  items     url/gdrive: [(file name, url or id)]; hf: repo id + patterns + rev (the exact snapshot we used)
  gb        approximate size on disk
  licence   short licence note; read the source's own terms before using or sharing
  used_by   experiments (plan: doc/2026_11_25/PLAN_evidence_pipeline_2026-11-25.md, kept outside git)
Nothing is unpacked: the experiment code reads the archives or unpacks what it needs into the workspace.
Dropped on 2026-11-25 and therefore absent: EgoTracks (needs an institutional Ego4D licence), SA-V (957 GB).
"""

PT = 'https://storage.googleapis.com/dm-perception-test/'
GD = 'https://drive.usercontent.google.com/download?export=download&confirm=t&id='

DATASETS = {
    # ---- domain 2: real photos, spatial relations ------------------------------------------------------------
    'vsr': dict(kind='hf', repo='cambridgeltl/vsr_random', rev='b2053328fafdd018ff56cf1dfa9643caaa4e69b8', patterns=None, gb=0.003,
                licence='CC BY 4.0 (annotations); images are COCO 2017 (Flickr terms)', used_by=['E1-d2', 'E2'],
                note='questions only; images come from coco2017'),
    'coco2017': dict(kind='url', gb=19.3, licence='annotations CC BY 4.0; images under Flickr terms of use',
                     used_by=['E1-d2', 'E2'], items=[('val2017.zip', 'http://images.cocodataset.org/zips/val2017.zip'),
                                                    ('train2017.zip', 'http://images.cocodataset.org/zips/train2017.zip')]),
    'whatsup': dict(kind='gdrive', gb=0.18, licence='MIT (code repo amitakamath/whatsup_vlms); see repo for data terms',
                    used_by=['E2'], items=[('controlled_images_dataset.json', '1ap8mmmpQjLIjPGuplkpBgc1hoEHCj4hm'),
                                           ('controlled_clevr_dataset.json', '1unNNosLbdy9NDjgj4l8fsQP3WiAAGA6z'),
                                           ('controlled_images.tar.gz', '19KGYVQjrV3syb00GgcavB2nZTW5NXX0H'),
                                           ('controlled_clevr.tar.gz', '13jdBpg8t3NqW3jrL6FK8HO93vwsUjDxG')]),
    'cv_bench': dict(kind='hf', repo='nyu-visionx/CV-Bench', rev='bc284db50d036958861cb60cdd7b77612052ce0d', patterns=None, gb=0.39, licence='Apache-2.0',
                     used_by=['E2']),
    'blink': dict(kind='hf', repo='BLINK-Benchmark/BLINK', rev='a3666eb249237ba3d5eca8db21176cc47967e040', patterns=['Relative_Depth/*', 'Multi-view_Reasoning/*', '*.md'],
                  gb=0.03, licence='Apache-2.0', used_by=['E2']),
    'tallyqa': dict(kind='url', gb=0.006, licence='see github.com/manoja328/TallyQA_dataset', used_by=['E2'],
                    items=[('tallyqa.zip', 'https://github.com/manoja328/tallyqa/blob/master/tallyqa.zip?raw=true')]),
    # ---- zoom / fine detail ---------------------------------------------------------------------------------
    'visual_cot': dict(kind='hf', repo='deepcs233/Visual-CoT', rev='223d2d8c1146fda2bb918801b8276c587b78b61c', patterns=['metadata/*', '*.md'], gb=0.17,
                       licence='Apache-2.0', used_by=['E2'],
                       note='question files only (the 143 GB image copy is NOT downloaded); images come from gqa and cub'),
    'gqa': dict(kind='url', gb=21.8, licence='CC BY 4.0', used_by=['E2'],
                items=[('images.zip', 'https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip')]),
    'cub': dict(kind='manual', gb=1.15, licence='research and non-commercial use only; do not re-share',
                used_by=['E2'], files=['CUB_200_2011.tgz'],
                how='Open https://www.vision.caltech.edu/datasets/cub_200_2011/ in a browser, download CUB_200_2011.tgz '
                    '(the server refuses scripts), and save it as <data>/cub/CUB_200_2011.tgz. Do not unpack.'),
    'vstar_bench': dict(kind='hf', repo='craigwu/vstar_bench', rev='d9ae62c903da0c98336e85c5ee89cd863b04b4da', patterns=None, gb=0.37, licence='see the dataset card',
                        used_by=['E2']),
    'hr_bench': dict(kind='hf', repo='DreamMr/HR-Bench', rev='83b9013d6293b85dc507e87199ca52517536939c', patterns=None, gb=9.2, licence='other: see its LICENSE file',
                     used_by=['E2']),
    # ---- text in images --------------------------------------------------------------------------------------
    'textvqa': dict(kind='url', gb=6.7, licence='CC BY 4.0; images from Open Images (CC BY 2.0)', used_by=['E2'],
                    items=[('TextVQA_0.5.1_train.json', 'https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_train.json'),
                           ('TextVQA_0.5.1_val.json', 'https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json'),
                           ('train_val_images.zip', 'https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip')]),
    'chartqa': dict(kind='hf', repo='ahmed-masry/ChartQA', rev='af8b6f5c08c95085271561c2a3f9d15f2b5a9031', patterns=None, gb=1.8, licence='GPL-3.0', used_by=['E2']),
    'docvqa': dict(kind='manual', gb=8.0, licence='RRC terms: each user registers; do not re-share', used_by=['E2'],
                   files=['spdocvqa_images.tar.gz', 'spdocvqa_qas.zip'],
                   how='Register at https://rrc.cvc.uab.es (DocVQA, Task 1 "Single Page"), accept the terms, download '
                       'spdocvqa_images.tar.gz and spdocvqa_qas.zip into <data>/docvqa/. Do not unpack.'),
    'infographicvqa': dict(kind='manual', gb=1.9, licence='RRC terms: each user registers; do not re-share',
                           used_by=['E2'], files=['infographicsvqa_images.tar.gz', 'infographicsvqa_qas.zip',
                                                  'infographicsvqa_ocr.tar.gz'],
                           how='Same RRC account (DocVQA, Task 3 "Infographics VQA"): download the images, question '
                               'and OCR archives into <data>/infographicvqa/. Do not unpack.'),
    # ---- video ----------------------------------------------------------------------------------------------
    'perception_test': dict(kind='url', gb=97.0, licence='CC BY 4.0', used_by=['E2'],
                            items=[(f.rsplit('/', 1)[1], PT + f) for f in (
                                'zip_data/mc_question_train_annotations.zip', 'zip_data/mc_question_valid_annotations.zip',
                                'misc/mc_question_train_id_list.csv', 'misc/mc_question_valid_id_list.csv',
                                'misc/cut_frame_mapping_train.json', 'misc/cut_frame_mapping_valid.json',
                                'zip_data/train_videos.zip', 'zip_data/valid_videos.zip')]),
    # ---- tools (model weights) ---------------------------------------------------------------------------------
    'models/Depth-Anything-V2-Small-hf': dict(kind='hf_model', repo='depth-anything/Depth-Anything-V2-Small-hf', rev='5426e4f0f36572d16453bbda7a8389317b1bef99',
                                              gb=0.1, licence='Apache-2.0 (Small model only)', used_by=['E1-d2', 'E2']),
    'models/grounding-dino-tiny': dict(kind='hf_model', repo='IDEA-Research/grounding-dino-tiny', rev='a2bb814dd30d776dcf7e30523b00659f4f141c71', gb=1.3,
                                       licence='Apache-2.0', used_by=['E1-d2', 'E2']),
}

# Named groups for --only: what one experiment needs
GROUPS = {
    'e1_d2': ['vsr', 'coco2017', 'models/Depth-Anything-V2-Small-hf', 'models/grounding-dino-tiny'],
    'e6a': ['perception_test', 'vsr', 'coco2017', 'chartqa', 'visual_cot', 'gqa'],
    'small': [k for k, v in DATASETS.items() if v['gb'] < 2 and v['kind'] != 'manual'],
}
