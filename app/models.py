from django.db import models
from django.contrib.auth.models import User

class Cart(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="cart")
    items = models.JSONField(default=dict)  # Use JSONField to store items and quantities

    def __str__(self):
        return f"Cart of {self.user.username}"

class Product(models.Model):
    name = models.CharField(max_length=255)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    image_url = models.URLField()  # Or use an ImageField for locally stored images

    def __str__(self):
        return self.name

