from django.contrib import admin
from mld.models import ML_Type, Dataset, Algorithm, Entry, Graph, File_Type, Parameter_Name, Data_Type

admin.site.register(ML_Type)
admin.site.register(Dataset)
admin.site.register(Algorithm)
admin.site.register(Entry)
admin.site.register(Graph)
admin.site.register(File_Type)
admin.site.register(Parameter_Name)
admin.site.register(Data_Type)