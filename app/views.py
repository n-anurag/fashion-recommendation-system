from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.core.files.storage import default_storage  # Import default_storage
import logging
import os
import pickle
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.preprocessing import image
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.contrib import messages


# Create a logger
logger = logging.getLogger(__name__)

# Set the logging level
logger.setLevel(logging.INFO)

# Create a file handler
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.INFO)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def home(request):
    logger.info('Home page accessed')
    return render(request, 'app/home.html')

def product_detail(request):
    logger.info('Product detail page accessed')
    return render(request, 'app/productdetail.html')

def cart(request, product_name):
    logger.info('Cart page accessed for product: %s', product_name)
    context = {'product_name': product_name}
    return render(request, 'app/cart.html', context)
def add_to_cart(request):
    # Initialize the cart if not already in the session
    if 'cart_items' not in request.session:
        request.session['cart_items'] = []

    if request.method == 'POST':
        # Add product to the cart
        product_name = request.POST.get('product_name')
        if product_name:
            # Avoid duplicates by checking if the product is already in the cart
            if product_name not in request.session['cart_items']:
                request.session['cart_items'].append(product_name)
                request.session.modified = True  # Mark session as modified to save changes
                logger.info('Added product to cart: %s', product_name)
        return redirect('add_to_cart')  # Redirect to the cart page after adding

    elif request.method == 'GET':
        # Render the cart page with cart items
        cart_items = request.session.get('cart_items', [])
        return render(request, 'app/addtocart.html', {'cart_items': cart_items})



def buy_now(request):
    logger.info('Buy now page accessed')
    return render(request, 'app/buynow.html')

def profile(request):
    logger.info('Profile page accessed')
    return render(request, 'app/profile.html')

def address(request):
    logger.info('Address page accessed')
    return render(request, 'app/address.html')

def orders(request):
    logger.info('Orders page accessed')
    return render(request, 'app/orders.html')

def change_password(request):
    logger.info('Change password page accessed')
    return render(request, 'app/changepassword.html')

def mobile(request):
    logger.info('Mobile page accessed')
    return render(request, 'app/mobile.html')
#This deals with authentication of login
def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        next_url = request.POST.get('next', '/')  # Default to home page if `next` is not provided

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            logger.info(f'User {username} logged in successfully.')

            # Redirect to `next` if provided, otherwise to default checkout or home
            if next_url:
                return redirect(next_url)
            return redirect('checkout')  # Default redirection
        else:
            messages.error(request, "Invalid credentials. Please try again.")
            return render(request, 'app/login.html', {'next': next_url})

    # For GET request, retain the `next` parameter
    next_url = request.GET.get('next', '/')
    logger.info('Login page accessed')
    return render(request, 'app/login.html', {'next': next_url})

def customerregistration(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')

        if password != confirm_password:
            messages.error(request, "Passwords do not match!")
            return render(request, 'app/customerregistration.html')

        try:
            # Create user
            user = User.objects.create_user(username=username, email=email, password=password)
            user.save()

            # Login the user after registration
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                logger.info(f'New user registered and logged in: {username}')
                messages.success(request, "Registration and login successful!")
                return redirect('login')  # Redirect to the login page after registration

        except Exception as e:
            logger.error(f"Error during registration: {e}")
            messages.error(request, "Error during registration!")
            return render(request, 'app/customerregistration.html')

    logger.info('Customer registration page accessed')
    return render(request, 'app/customerregistration.html')

def remove_from_cart(request, product_name):
    # Ensure 'cart_items' exists in the session
    if 'cart_items' in request.session:
        cart_items = request.session['cart_items']
        # Check if the product exists in the cart
        if product_name in cart_items:
            del cart_items[product_name]  # Remove the product from the dictionary
            request.session['cart_items'] = cart_items  # Save updated cart
            request.session.modified = True  # Mark session as modified
            logger.info(f'Removed product from cart: {product_name}')
        else:
            logger.warning(f'Product {product_name} not found in cart')
    else:
        logger.warning('Cart is empty or does not exist in session')

    # Redirect back to the cart page
    return redirect('add_to_cart')


from django.http import JsonResponse

def update_cart(request):
    if request.method == "POST":
        product_name = request.POST.get('product_name')
        action = request.POST.get('action')  # 'add' or 'subtract'
        cart = request.session.get('cart_items', [])
        prices = request.session.get('prices', {})  # Store prices for products in session

        if product_name and action:
            # Update the cart based on action
            if action == "add":
                prices[product_name] = prices.get(product_name, 110)  # Default price Rs. 110
                cart.append(product_name)
            elif action == "subtract" and product_name in cart:
                cart.remove(product_name)

            # Save updated cart and prices in session
            request.session['cart_items'] = cart
            request.session['prices'] = prices
            request.session.modified = True

            # Calculate the new summary
            total_items = len(cart)
            subtotal = sum(prices.get(item, 110) for item in cart)  # Sum up all item prices
            shipping = 70
            total = subtotal + shipping

            # Return updated summary as JSON
            return JsonResponse({
                'total_items': total_items,
                'subtotal': subtotal,
                'shipping': shipping,
                'total': total,
            })
    return JsonResponse({'error': 'Invalid request'}, status=400)


def checkout(request):
    cart_items = request.session.get('cart_items', [])
    cart_quantities = request.session.get('cart_quantities', {})
    total_price = 0
    items_with_details = []

    for product in cart_items:
        quantity = cart_quantities.get(product, 1)
        price_per_item = 110  # Replace with your product price logic
        total_price += price_per_item * quantity
        items_with_details.append({
            'product': product,
            'quantity': quantity,
            'price': price_per_item * quantity
        })

    return render(request, 'app/checkout.html', {
        'items_with_details': items_with_details,
        'total_price': total_price,
        'shipping': 70,
        'grand_total': total_price + 70
    })

def clear_cart(request):
    """Clears all items from the shopping cart."""
    if 'cart_items' in request.session:
        request.session['cart_items'] = []  # Clear all items
        request.session['cart_quantities'] = {}  # Clear all quantities if used
        request.session.modified = True  # Mark session as modified
    return redirect('add_to_cart')  # Redirect back to the cart page

def index(request):
    if request.method == "POST" and request.FILES.get('upload'):
        upload = request.FILES['upload']
        logger.info('Image uploaded successfully')
        return render(request, 'base.html', {'message': 'Image uploaded successfully'})

def fetch(request):
    if request.method == "POST":
        if 'upload' not in request.FILES:
            err = 'No images Selected'
            logger.error(err)
            return render(request, './app/final.html', {'err': err})

        upload = request.FILES['upload']
        filename = upload.name
        with default_storage.open(filename, 'wb+') as destination:
            for chunk in upload.chunks():
                destination.write(chunk)

        try:
            feature_list = np.array(pickle.load(open('./shoppinglyx/embeddings.pkl', 'rb')))
            filenames = pickle.load(open('./shoppinglyx/filenames.pkl', 'rb'))
        except Exception as e:
            logger.error('Error loading pickle files: %s', e)
            return render(request, './app/final.html', {'err': f"Error loading pickle files: {e}"})

        model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        model.trainable = False
        model = Sequential([model, GlobalMaxPooling2D()])

        img = image.load_img(os.path.realpath(destination.name), target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)

        neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
        neighbors.fit(feature_list)
        distances, indices = neighbors.kneighbors([normalized_result])

        result_urls = []
        for file in indices[0][0:5]:
            image_name = filenames[file].split("/")[-1]
            result_urls.append(image_name)

        uploaded_image_url = f"{settings.MEDIA_URL}{filename}"
        logger.info('Image processed successfully')
        return render(request, './app/final.html', {'context': result_urls, 'uploaded_image_url': uploaded_image_url})

def final(request):
    logger.info('Final page accessed')
    return render(request, './app/final.html')

def logout_view(request):
    logger.info('User logged out')
    logout(request)
    return redirect('home')