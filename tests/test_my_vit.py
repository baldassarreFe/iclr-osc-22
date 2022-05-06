import timm
import torch

from osc.models.vit import MyVisionTransformer


def test_pretrained_vit():
    x = torch.rand(2, 3, 224, 224)

    pre = timm.create_model("vit_small_patch8_224_dino", pretrained=True)
    pre.eval()

    y_pre = pre.forward_features(x)[:, 1:, :]

    vit = MyVisionTransformer(
        patch_size=8, depth=12, num_heads=6, embed_dim=384, output_norm=True
    )
    vit.load_state_dict(pre.state_dict())
    vit.eval()

    _, y_vit = vit.forward(x)

    torch.testing.assert_allclose(y_vit.flatten(), y_pre.flatten())


def test_vit_small_image():
    dim = 32
    patch_size = 16
    vit = MyVisionTransformer(
        patch_size=patch_size, depth=2, num_heads=4, embed_dim=dim, output_norm=True
    )
    for size in [224, 96]:
        x = torch.rand(2, 3, size, size)
        g, y = vit.forward(x)
        assert g.shape == (x.shape[0], dim)
        assert y.shape == (x.shape[0], size // patch_size, size // patch_size, dim)
