from django import forms
from django.core.exceptions import ValidationError
from django.core import validators
from dl.models import Gallery, Category, Image

from django.core.files.uploadedfile import InMemoryUploadedFile
# from ads.humanize import naturalsize
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
    # max_upload_limit = 2 * 1024 * 1024
    # max_upload_limit_text = naturalsize(max_upload_limit)

    # Call this 'picture' so it gets copied from the form to the in-memory model
    # It will not be the "bytes", it will be the "InMemoryUploadedFile"
    # because we need to pull out things like content_type
    image = forms.FileField(required=False, label='Image to Upload <= ')
    upload_field_name = 'image'

    class Meta:
        model = Image
        fields = ['gallery', 'category', 'title', 'description', 'image']

    # Validate the size of the picture
    def clean(self):
        cleaned_data = super().clean()
        img = cleaned_data.get('image')
        if img is None:
            return
        # if len(img) > self.max_upload_limit:
        #     self.add_error('picture', "File must be < "+self.max_upload_limit_text+" bytes")

    # Convert uploaded File object to a picture
    def save(self, commit=True):
        instance = super(CreateImageForm, self).save(commit=False)

        # We only need to adjust picture if it is a freshly uploaded file
        f = instance.image   # Make a copy
        if isinstance(f, InMemoryUploadedFile):  # Extract data from the form to the model
            bytearr = f.read()
            instance.content_type = f.content_type
            instance.image = bytearr  # Overwrite with the actual image data

        if commit:
            instance.save()

        return instance
