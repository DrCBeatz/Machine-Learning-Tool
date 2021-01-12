from django import forms
from django.core.exceptions import ValidationError
from django.core import validators
from mld.models import ML_Type, Dataset, Algorithm, Entry, Graph, File_Type, Parameter_Name
from django.core.files.uploadedfile import InMemoryUploadedFile
# from ads.humanize import naturalsize
from django.core import serializers



class CreateForm(forms.ModelForm):

    class Meta:
        model = Entry
        fields = ['ml_type', 'dataset', 'algorithm', 'parameter_1_value', 'parameter_2_value', 'parameter_3_value']

class CreateDatasetForm(forms.ModelForm):

    data = forms.FileField(required=False, label='File to Upload')
    upload_field_name = 'data'

    class Meta:
        model = Dataset
        fields = ['title', 'description', 'ml_type', 'data', 'file_type', 'has_header', 'public', 'target_datatype']

    def save(self, commit=True):
        instance = super(CreateDatasetForm, self).save(commit=False)

        # We only need to adjust data if it is a freshly uploaded file
        f = instance.data   # Make a copy
        if isinstance(f, InMemoryUploadedFile):  # Extract data from the form to the model
            bytearr = f.read()
            instance.content_type = f.content_type
            instance.data = bytearr  # Overwrite with the actual data

        if commit:
            instance.save()

        return instance

class CreateGraphForm(forms.ModelForm):

    class Meta:
        model = Graph
        fields = ['ml_type', 'dataset', 'algorithm', 'parameter_to_plot']