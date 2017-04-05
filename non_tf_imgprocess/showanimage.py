from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000

mosfile = "D:/Iheya_n/NBC_2m_2014/mosaics/NBC_2m_2014_labelled_layers.tif"
img = Image.open(mosfile)
x = 386
y = 11698
img1 = img.crop(
        (
                x-25,
                y-25,
                x+25,
                y+25
        )
    )
img1.show()
