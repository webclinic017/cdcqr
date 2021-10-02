import matplotlib.pyplot as plt
import base64
from io import BytesIO

dpi=100

def plotstart():
    img=BytesIO()
    plt.clf()
    return img


def plotend(img):
    plt.savefig(img,format='png',dpi=dpi,bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

