import torch
from torchvision import transforms
import clip


def test_cuda():
    device = "cuda"
    cutn = 16
    shape = (256, 256)
    lr = 0.03
    steps = 500
    clip_model = "ViT-B/32"
    prompt = "a rainbow flower"

    image = torch.rand((1, 3, shape[0], shape[1]), device=device, requires_grad=True)

    opt = torch.optim.Adam((image,), lr)

    f = transforms.Compose(
        [
            lambda x: torch.clamp((x + 1) / 2, min=0, max=1),
            transforms.RandomAffine(degrees=60, translate=(0.1, 0.1)),
            transforms.RandomGrayscale(p=0.2),
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            transforms.Resize(224),
        ]
    )

    m = clip.load(clip_model, jit=False)[0].eval().requires_grad_(False).to(device)

    embedding = m.encode_text(clip.tokenize(prompt).to(device))

    def total_variation_loss(img):
        yv = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
        xv = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
        return (yv + xv) / (1 * 3 * shape[0] * shape[1])

    def spherical_distance_loss(x, y):
        return (
            (
                torch.nn.functional.normalize(x, dim=-1)
                - torch.nn.functional.normalize(y, dim=-1)
            )
            .norm(dim=-1)
            .div(2)
            .arcsin()
            .pow(2)
            .mul(2)
            .mean()
        )

    for i in range(steps):
        opt.zero_grad()
        clip_in = m.encode_image(
            torch.cat([f(image.add(1).div(2)) for _ in range(cutn)])
        )
        loss = (
            spherical_distance_loss(clip_in, embedding.unsqueeze(0))
            + (image - image.clamp(-1, 1)).pow(2).mean() / 2
            + total_variation_loss(image)
        )
        loss.backward()
        opt.step()

    img = transforms.ToPILImage()(image.squeeze(0).clamp(-1, 1) / 2 + 0.5)
    img.save("images/horde_basic_cuda.webp", quality=90)
