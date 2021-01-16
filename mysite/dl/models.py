from django.db import models
from django.conf import settings
from django.core.validators import MinLengthValidator

from imagekit.models import ImageSpecField
from imagekit.processors import ResizeToFill, Transpose

class Gallery(models.Model):
    title = models.CharField(
            max_length=200,
            validators=[MinLengthValidator(2, "Title must be greater than 2 characters")]
    )
    description = models.CharField(max_length=2000, null=True, blank=True)
    thumbnail = models.BinaryField(null=True, blank=True, editable=True)
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    def __str__(self):
        return self.title

class Category(models.Model):
    title = models.CharField(
            max_length=200,
            validators=[MinLengthValidator(2, "Title must be greater than 2 characters")]
    )
    description = models.CharField(max_length=2000, null=True, blank=True)
    gallery = models.ForeignKey(Gallery, on_delete=models.CASCADE)
    thumbnail = models.BinaryField(null=True, blank=True, editable=True)
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    def __str__(self):
        return self.title


class Image(models.Model):
    title = models.CharField(
            max_length=200,
            validators=[MinLengthValidator(2, "Title must be greater than 2 characters")]
    )
    description = models.CharField(max_length=2000, null=True, blank=True)
    gallery = models.ForeignKey(Gallery, on_delete=models.CASCADE)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    content_type = models.CharField(max_length=256, null=True, blank=True,
                                    help_text='The MIMEType of the file')
    image = models.ImageField(upload_to="images", null=True, blank=True)
    # image = models.ImageField(upload_to="images", default="")
    thumbnail = ImageSpecField(source='image', processors=[Transpose(), ResizeToFill(600, 400)], format='JPEG', options={'quality':60})
    # processors.Thumbnail(width=72, height=72, crop=True)
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title
