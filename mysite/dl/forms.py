from django import forms
from django.core.exceptions import ValidationError
from django.core import validators
from dl.models import Gallery, Category, Image

from django.core.files.uploadedfile import InMemoryUploadedFile
#from django.core import serializers

class CreateGalleryForm(forms.ModelForm):
    class Meta:
        model = Gallery
        fields = ['title', 'description']

class CreateCategoryForm(forms.ModelForm):
    class Meta:
        model = Category
        fields = ['gallery', 'title', 'description']

class CreateImageForm(forms.ModelForm):

    # def __init__(self, *args, **kwargs):
    #     super(CreateImageForm, self).__init__(*args, **kwargs)
    #     for visible in self.visible_fields():
    #         visible.field.widget.attrs['class'] = 'form-control'
    class Meta:
        model = Image
        fields = ['gallery', 'category', 'title', 'description', 'image']
