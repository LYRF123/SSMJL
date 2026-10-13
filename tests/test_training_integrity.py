"""Regression checks for pretrained weights, paired samples and validation labels."""

import importlib.util
import sys
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Finetune'))

from ssmjl_utils import prepare_legacy_timm, validate_bands

prepare_legacy_timm()
from feature_extractor import ConvMAEFeatureExtractor
from finetune_classifier_head import ImageDataset, ViTClassifier
from train_dual_classifier import DualEncoderClassifier, DualSourceDataset, SupConModel, Trainer, load_supcon
import models_convmae


def write_tif(path, channels=5):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, 'w', driver='GTiff', width=20, height=16,
                       count=channels, dtype='float32') as image:
        image.write(np.full((channels, 16, 20), 0.5, dtype=np.float32))


@pytest.mark.parametrize('bands', [[], [-1], [0, 0]])
def test_invalid_band_selection_fails(bands):
    with pytest.raises(ValueError):
        validate_bands(bands)


def test_repeated_stems_in_different_classes_do_not_overwrite_pairs(tmp_path):
    for treatment in ('ht', 'ck'):
        for label in ('0', '1'):
            write_tif(tmp_path / treatment / label / 'sample.tiff')
    dataset = DualSourceDataset(tmp_path / 'ht', tmp_path / 'ck', range(5))
    assert len(dataset) == 2
    assert [pair[2] for pair in dataset.pairs] == [0, 1]
    ht_cm, ck_cm, ht_sc, ck_sc, _, _ = dataset[0]
    assert all(image.shape == (5, 224, 224) for image in (ht_cm, ck_cm, ht_sc, ck_sc))


def test_unmatched_pair_is_reported(tmp_path):
    write_tif(tmp_path / 'ht' / '0' / 'sample.tif')
    (tmp_path / 'ck' / '0').mkdir(parents=True)
    with pytest.raises(ValueError, match='Unmatched HT/CK'):
        DualSourceDataset(tmp_path / 'ht', tmp_path / 'ck', range(5))


def test_duplicate_tif_and_tiff_stem_is_reported(tmp_path):
    for treatment in ('ht', 'ck'):
        write_tif(tmp_path / treatment / '0' / 'sample.tif')
    write_tif(tmp_path / 'ht' / '0' / 'sample.tiff')
    with pytest.raises(ValueError, match='Duplicate'):
        DualSourceDataset(tmp_path / 'ht', tmp_path / 'ck', range(5))


def test_pair_validation_reuses_training_class_indices(tmp_path):
    for treatment in ('ht', 'ck'):
        write_tif(tmp_path / treatment / '1' / 'sample.tif')
    dataset = DualSourceDataset(tmp_path / 'ht', tmp_path / 'ck', range(5),
                                class_to_idx={'0': 0, '1': 1, '2': 2})
    assert dataset.pairs[0][2] == 1
    assert dataset.num_classes == 3


def test_single_image_validation_reuses_training_indices_and_reads_tiff(tmp_path):
    write_tif(tmp_path / '1' / 'sample.tiff')
    dataset = ImageDataset(tmp_path, range(5), class_to_idx={'0': 0, '1': 1, '2': 2})
    image, label = dataset[0]
    assert label == 1
    assert image.shape == (5, 224, 224)


def test_unknown_validation_class_fails(tmp_path):
    write_tif(tmp_path / 'new' / 'sample.tif')
    with pytest.raises(ValueError, match='Unknown validation'):
        ImageDataset(tmp_path, range(5), class_to_idx={'0': 0})


def test_corrupt_image_does_not_become_a_zero_sample(tmp_path):
    folder = tmp_path / '0'
    folder.mkdir()
    (folder / 'corrupt.tif').write_text('not a TIFF')
    dataset = ImageDataset(tmp_path, range(5))
    with pytest.raises(RuntimeError, match='corrupt.tif'):
        dataset[0]


def test_actual_supcon_pretraining_checkpoint_loads_into_both_classifiers(tmp_path):
    source_path = ROOT / 'Pretrain/SupCon/10band_Supervised.py'
    sys.path.insert(0, str(source_path.parent))
    spec = importlib.util.spec_from_file_location('supcon_pretraining', source_path)
    pretraining = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pretraining)
    augmented = pretraining.AugmentationPipeline(5)(torch.rand(5, 32, 32))
    assert augmented.shape == (5, 224, 224)
    assert torch.isfinite(augmented).all()
    source = pretraining.FeatureExtractorWithProjection(
        backbone='vit', in_chans=5, pretrained=False, use_imagenet_pretrained=False)
    checkpoint = tmp_path / 'supcon.pth'
    torch.save({'state_dict': {'module.' + key: value for key, value in source.state_dict().items()}}, checkpoint)
    dual = load_supcon(checkpoint, torch.device('cpu'))
    single = ViTClassifier(in_chans=5, pretrained_backbone_path=checkpoint)
    for key, value in source.feature_extractor.state_dict().items():
        assert torch.equal(dual.encoder.state_dict()[key], value)
        assert torch.equal(single.feature_extractor.state_dict()[key], value)
    for key, value in source.projection_head.net.state_dict().items():
        assert torch.equal(dual.projection.state_dict()[key], value)
    source.eval()
    image = torch.rand(1, 5, 224, 224)
    with torch.no_grad():
        torch.testing.assert_close(source(image), dual(image))


def test_incomplete_supcon_checkpoint_fails(tmp_path):
    weights = SupConModel().state_dict()
    weights.pop('encoder.vit.blocks.0.attn.qkv.weight')
    checkpoint = tmp_path / 'incomplete.pth'
    torch.save(weights, checkpoint)
    with pytest.raises(ValueError, match='missing keys'):
        load_supcon(checkpoint, torch.device('cpu'))


def test_missing_checkpoint_path_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        ViTClassifier(in_chans=5, pretrained_backbone_path=tmp_path / 'missing.pth')


def test_convmae_restores_encoder_and_trained_100d_projection(tmp_path):
    source = models_convmae.convmae_convvit_tiny_patch16(in_chans=5)
    nn.init.constant_(source.decoder_embed[0].weight, 0.123)
    nn.init.constant_(source.decoder_embed[0].bias, 0.456)
    checkpoint = tmp_path / 'convmae.pth'
    torch.save({'model': source.state_dict()}, checkpoint)
    loaded = ConvMAEFeatureExtractor(checkpoint_path=checkpoint, device='cpu')
    assert torch.equal(loaded.projection.weight, source.decoder_embed[0].weight)
    assert torch.equal(loaded.projection.bias, source.decoder_embed[0].bias)
    for key, value in source.state_dict().items():
        if key == 'pos_embed' or key.startswith(('patch_embed', 'blocks', 'norm.')):
            assert torch.equal(loaded.convmae.state_dict()[key], value)
    with torch.no_grad():
        assert loaded.projection(loaded.forward_encoder(torch.rand(1, 5, 224, 224))).shape == (1, 100)


def test_convmae_checkpoint_missing_projection_fails(tmp_path):
    source = models_convmae.convmae_convvit_tiny_patch16(in_chans=5)
    weights = source.state_dict()
    weights.pop('decoder_embed.0.weight')
    checkpoint = tmp_path / 'missing_projection.pth'
    torch.save({'model': weights}, checkpoint)
    with pytest.raises(ValueError, match='decoder_embed.0.weight'):
        ConvMAEFeatureExtractor(checkpoint_path=checkpoint, device='cpu')


def test_frozen_encoders_remain_in_eval_during_classifier_training():
    convmae = nn.Module()
    convmae.forward_encoder = lambda image: convmae.dropout(image)
    convmae.dropout = nn.Dropout(0.8)
    convmae.projection = nn.Linear(4, 100)
    supcon = nn.Sequential(nn.Dropout(0.8), nn.Linear(4, 100))
    model = DualEncoderClassifier(convmae, supcon, dropout_rate=0)
    model.train()
    image = torch.rand(2, 4)
    logits = model(image, image, image, image)
    logits.sum().backward()
    assert model.classifier.training
    assert not convmae.training and not supcon.training
    assert all(parameter.grad is None for encoder in (convmae, supcon) for parameter in encoder.parameters())
    assert all(parameter.grad is not None for parameter in model.classifier.parameters())
    torch.testing.assert_close(logits, model(image, image, image, image))


def test_contrastive_loss_is_scale_invariant_and_handles_no_positive_pair():
    spec = importlib.util.spec_from_file_location('contrastive_loss', ROOT / 'Pretrain/SupCon/loss/supconloss.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    criterion = module.SupConLoss()
    features = torch.randn(4, 2, 100, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    loss = criterion(features, labels)
    torch.testing.assert_close(loss, criterion(features * 100, labels))
    loss.backward()
    assert torch.isfinite(features.grad).all()
    singleton = torch.randn(1, 1, 100, requires_grad=True)
    result = criterion(singleton, torch.tensor([0]))
    assert torch.isfinite(result)
    result.backward()
    assert torch.isfinite(singleton.grad).all()


def test_dual_report_restores_the_best_epoch_including_zero_accuracy(tmp_path, monkeypatch):
    model = nn.Module()
    model.classifier = nn.Linear(1, 1)
    trainer = Trainer(model, None, None, torch.device('cpu'),
                      {'lr': 0.01, 'wd': 0, 'epochs': 2, 'output_dir': str(tmp_path)})
    epochs = iter([0.0, 0.0])
    step = [0]
    def epoch(loader, training):
        if training:
            step[0] += 1
            nn.init.constant_(model.classifier.weight, step[0])
            trainer.optimizer.step()
            return 1, 0, [], []
        return 1, next(epochs, 0), [int(model.classifier.weight.item())], [0]
    captured = []
    monkeypatch.setattr(trainer, '_epoch', epoch)
    monkeypatch.setattr(trainer, '_report', lambda predictions, labels: captured.extend(predictions))
    trainer.train()
    assert (tmp_path / 'best_model.pth').is_file()
    assert captured == [1]


@pytest.mark.parametrize('module_name', ['main_pretrain', 'main_finetune'])
def test_mcmae_labels_ignore_metadata_and_support_more_than_five_bands(tmp_path, module_name):
    for label in ('0', '2'):
        write_tif(tmp_path / 'train' / label / 'sample.tif', channels=7)
    write_tif(tmp_path / 'val' / '2' / 'sample.tif', channels=7)
    (tmp_path / 'train' / '00-metadata.txt').write_text('metadata')
    code = textwrap.dedent(f'''
        import sys
        sys.path.insert(0, {str(ROOT / 'Pretrain/MCMAE-Multispectral')!r})
        module = __import__({module_name!r})
        train = module.MultibandDataset({str(tmp_path / 'train')!r}, in_chans=7)
        val = module.MultibandDataset({str(tmp_path / 'val')!r}, in_chans=7, class_to_idx=train.class_to_idx)
        assert train.labels == [0, 1], train.labels
        assert val.labels == [1], val.labels
        assert len(train.mean) == len(train.std) == 7
        assert train.open_image(train.img_paths[0]).shape == (16, 20, 7)
    ''')
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('folder', ['Finetune', 'Pretrain/MCMAE-Multispectral'])
def test_tiny_convvit_architecture_matches_pretraining_checkpoint(folder):
    code = textwrap.dedent(f'''
        import sys
        sys.path.insert(0, {str(ROOT)!r})
        sys.path.insert(0, str(__import__('pathlib').Path({str(ROOT)!r}) / {folder!r}))
        from ssmjl_utils import prepare_legacy_timm
        prepare_legacy_timm()
        import models_convmae, models_convvit
        pretrain = models_convmae.convmae_convvit_tiny_patch16(in_chans=5)
        classifier = models_convvit.convvit_tiny_patch16(in_chans=5, num_classes=3)
        result = classifier.load_state_dict(pretrain.state_dict(), strict=False)
        assert set(result.missing_keys) == {{'head.weight', 'head.bias'}}, result
    ''')
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('engine_name', ['engine_pretrain', 'engine_finetune'])
def test_final_incomplete_accumulation_group_updates_parameters(engine_name):
    code = textwrap.dedent(f'''
        import sys
        from types import SimpleNamespace
        sys.path.insert(0, {str(ROOT)!r})
        sys.path.insert(0, {str(ROOT / 'Pretrain/MCMAE-Multispectral')!r})
        from ssmjl_utils import prepare_legacy_timm
        prepare_legacy_timm()
        import torch
        engine = __import__({engine_name!r})
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
            def forward(self, x, mask_ratio=None):
                output = self.weight * x
                return (output.square().mean(), None, None) if mask_ratio is not None else output
        class Scaler:
            def __call__(self, loss, optimizer, update_grad=True, **kwargs):
                loss.backward()
                if update_grad:
                    optimizer.step()
        model = Model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        data = [(torch.ones(1, 1), torch.zeros(1, 1))] * 2
        args = SimpleNamespace(accum_iter=3, lr=0.1, min_lr=0.0, warmup_epochs=0, epochs=1, mask_ratio=0.75)
        if {engine_name!r} == 'engine_pretrain':
            engine.train_one_epoch(model, data, optimizer, torch.device('cpu'), 0, Scaler(), args=args)
        else:
            engine.train_one_epoch(model, torch.nn.MSELoss(), data, optimizer, torch.device('cpu'), 0, Scaler(), args=args)
        torch.testing.assert_close(model.weight, torch.tensor([0.8]))
    ''')
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
