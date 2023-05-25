# make_index_all_models.py
# Create an index of the all model showcase (see examples.run_all_models)
import datetime
import glob
import os

TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>hordelib</title>
  <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Roboto&display=swap">
  <style>
    body {
        background-color: #333;
        color: #fff;
    }
    a {
        color: #fff;
        text-decoration: none;
    }
    .gallery {
        display: flex;
        flex-wrap: wrap;
        /*gap: 10px;*/
    }
    h1 {
        font-family: 'Roboto', sans-serif;
        font-size: 24px;
        text-align: left;
        margin-bottom: 20px;
    }
    p {
        font-family: 'Roboto', sans-serif;
        font-size: 14px;
        text-align: left;
        margin-bottom: 20px;
    }
    h2 {
        font-family: 'Roboto', sans-serif;
        font-size: 14px;
        text-align: left;
        margin-bottom: 20px;
    }
    .thumbnail {
        max-width: 160px;
        cursor: pointer;
    }
    figure {
        max-width: 160px;
        text-align: center;
        margin: 0 5px;
    }
    figcaption {
        font-family: 'Roboto', sans-serif;
        font-size: 12px;
        color: #fff;
        word-wrap: break-word; /* Force text to wrap onto multiple lines */
    }
  </style>
</head>
<body>
  <h1>Test of Each Model</h1>
  <h2>[timestamp] (UTC)</h2>
  <p><i>Note that this image set is <b>not</b> updated automatically as part of the unit tests.
  It has periodic manual updates due to the GPU time required.</i></p>
  <div class="gallery">
    [images]
  </div>
  <h1>Prompt and Settings</h1>
  <p>"a woman closeup made out of metal, (cyborg:1.1), realistic skin, detailed wire:1.3), (intricate details),
  hdr, (intricate details, hyperdetailed:1.2), cinematic shot, vignette, centered"</p>
  <p>Seed: 3688490319, Karras, 512x512, k_euler, 30 steps, CFG 6.5</p>
</body>
</html>
"""


def href(filename):
    return (
        f'<a href="{filename}"><figure><img class="thumbnail" src="{filename}">'
        f'<figcaption>{filename.replace(".png", "")}</figcaption></figure></a>'
    )


def create_index():
    # Get a list of images
    files = glob.glob("images/all_models/*.png")
    refs = []
    for imagefile in files:
        filename = os.path.basename(imagefile)
        refs.append(href(filename))
    in_refs = []

    # Poor man's template engine :)
    refs.sort()
    indexhtml = TEMPLATE.replace("[images]", "".join(refs))
    indexhtml = indexhtml.replace("[input_images]", "".join(in_refs))
    indexhtml = indexhtml.replace(
        "[timestamp]",
        datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    )

    # Output results
    with open("images/all_models/index.html", "w") as outfile:
        outfile.write(indexhtml)


if __name__ == "__main__":
    create_index()
