# make_index.py
# This isn't a test.
# It's a 30 second hack to create an index.html that shows all our tests results.
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
  <h1>Latest Build Results</h1>
  <h2>[timestamp] (UTC)</h2>
  <div class="gallery">
    [images]
  </div>
  <h1>Input files for the tests</h1>
  <div class="gallery">
    [input_images]
  </div>
</body>
</html>
"""


def href(filename):
    return f'<a href="{filename}"><figure><img class="thumbnail" src="{filename}"><figcaption>{filename.replace("_", " ")}</figcaption></figure></a>'


def create_index():
    # Get a list of images
    files = glob.glob("images/*.png")
    input_files = glob.glob("images/test_*")
    refs = []
    for imagefile in files:
        filename = os.path.basename(imagefile)
        refs.append(href(filename))
    in_refs = []
    for imagefile in input_files:
        filename = os.path.basename(imagefile)
        in_refs.append(href(filename))

    # Poor man's template engine :)
    refs.sort()
    indexhtml = TEMPLATE.replace("[images]", "".join(refs))
    indexhtml = indexhtml.replace("[input_images]", "".join(in_refs))
    indexhtml = indexhtml.replace(
        "[timestamp]",
        datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    )

    # Output results
    with open("images/index.html", "w") as outfile:
        outfile.write(indexhtml)


if __name__ == "__main__":
    create_index()
