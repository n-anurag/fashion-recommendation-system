from django.shortcuts import render
import pickle
import tensorflow
import numpy as np
from numpy.linalg import norm


from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans


from .form import ImageForm
import os
import cv2
from django.core.files.storage import FileSystemStorage
from django.core.files.storage import default_storage



def home(request):
 return render(request, 'app/home.html')

def product_detail(request):
 return render(request, 'app/productdetail.html')

def add_to_cart(request):
 return render(request, 'app/addtocart.html')

def buy_now(request):
 return render(request, 'app/buynow.html')

def profile(request):
 return render(request, 'app/profile.html')

def address(request):
 return render(request, 'app/address.html')

def orders(request):
 return render(request, 'app/orders.html')

def change_password(request):
 return render(request, 'app/changepassword.html')

def mobile(request):
 return render(request, 'app/mobile.html')

def login(request):
 return render(request, 'app/login.html')

def customerregistration(request):
 return render(request, 'app/customerregistration.html')

def checkout(request):
 return render(request, 'app/checkout.html')


def fetch_recommendations(request):
    # Example: System generates 5 recommended images dynamically
    recommended_images = ['10030.jpg', '7607.jpg', '9550.jpg', '12589.jpg', '14013.jpg']

    # Create a context list where each image includes its path and name
    context = [
        {"image_path": f"dataset/{filename}", "name": filename}  # dataset is the folder where images are stored
        for filename in recommended_images
    ]

    return render(request, 'your_template.html', {"context": context})


def index(request):
    if request.method =="POST" and request.FILES['upload']:
        if'upload' not in request.FILES:
            err='No images Selected'
            return render(request,'base.html',{'err':err})
        f=request.FILES['upload']
        if f=='':
            wee='No files selected'
            return render(request,'base.html',{'err':err})
        upload =request.FILES['upload']
        
        #file_url=


        # image = load_img(file_url, target_size=(224, 224))
        # numpy_array = img_to_array(image)
        # image_batch = np.expand_dims(numpy_array, axis=0)
        # processed_image =ResNet50,.preprocess_input(image_batch.copy())

def fetch(request):
    if request.method == "POST":
        if 'upload' not in request.FILES:
            err = 'No images Selected'
            return render(request, './app/final.html', {'err': err})
        
        upload = request.FILES['upload']
        filename = upload.name

        # Save the uploaded image
        with default_storage.open(filename, 'wb+') as destination:
            for chunk in upload.chunks():
                destination.write(chunk)

        # Load precomputed data
        feature_list = np.array(pickle.load(open('./shoppinglyx/embeddings.pkl', 'rb')))
        filenames = pickle.load(open('./shoppinglyx/filenames.pkl', 'rb'))
        cluster_assignments = pickle.load(open('./shoppinglyx/cluster_assignments.pkl', 'rb'))
        cluster_centroids = pickle.load(open('./shoppinglyx/cluster_centroids.pkl', 'rb'))

        # Initialize ResNet50 model for feature extraction
        model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        model.trainable = False
        model = Sequential([model, GlobalMaxPooling2D()])

        # Preprocess the uploaded image
        img = image.load_img(os.path.realpath(destination.name), target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)

        # Find the closest cluster for the input image
        distances_to_centroids = np.linalg.norm(cluster_centroids - normalized_result, axis=1)
        closest_cluster = np.argmin(distances_to_centroids)

        # Get all images in the closest cluster
        cluster_images = [filenames[i] for i in range(len(cluster_assignments)) if cluster_assignments[i] == closest_cluster]

        # Select the top 5 images to recommend
        recommended_images = cluster_images[:5]

        # Prepare URLs for recommended images
        result_urls = [img.split('/')[-1] for img in recommended_images]
        print(result_urls)

        return render(request, './app/final.html', {'context': result_urls})


def final(request):
    return render(request,'./app/final.html')