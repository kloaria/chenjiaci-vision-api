import chromadb
import torch

from PIL import Image
from io import BytesIO

from fastapi import FastAPI, UploadFile, File

from transformers import CLIPProcessor, CLIPModel


app = FastAPI()



# =====================
# Chroma连接
# =====================

client = chromadb.CloudClient(

    api_key=os.environ["CHROMA_API_KEY"],

    tenant=os.environ["CHROMA_TENANT"],

    database=os.environ["CHROMA_DATABASE"]

)


collection = client.get_collection(
    "chenjiaci_exhibits"
)



# =====================
# CLIP加载
# =====================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
)


processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)


model.to(device)

model.eval()



# =====================
# 图片转embedding
# =====================

def get_embedding(image):


    inputs = processor(
        images=image,
        return_tensors="pt"
    )


    inputs = {
        k:v.to(device)
        for k,v in inputs.items()
    }


    with torch.no_grad():

        outputs = model.get_image_features(
            **inputs
        )


    if hasattr(outputs,"image_embeds"):

        feature = outputs.image_embeds


    elif hasattr(outputs,"pooler_output"):

        feature = outputs.pooler_output


    else:

        feature = outputs



    feature = feature / feature.norm(
        dim=-1,
        keepdim=True
    )


    return feature.cpu().numpy()[0].tolist()



# =====================
# 图片搜索接口
# =====================

@app.post("/search")
async def search_image(
        file:UploadFile=File(...)
):


    # 读取图片

    image_bytes = await file.read()


    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")



    # embedding

    embedding = get_embedding(
        image
    )



    # Chroma查询

    result = collection.query(

        query_embeddings=[
            embedding
        ],

        n_results=5

    )



    return {

        "results":
        result["metadatas"][0]

    }