from django.urls import path, reverse_lazy
from . import views
from django.views.generic import TemplateView

app_name = 'dl'
urlpatterns = [

    # path('', views.DlHomeView.as_view(), name='dl_home'),
    path('about/', views.DlAboutView.as_view(), name='dl_about'),


    # path('images', views.ImageListView.as_view(), name='all'),
    path('', views.ImageListView.as_view(), name='all'),
    path('images/<int:pk>', views.ImageDetailView.as_view(), name='image_detail'),
    path('images/create', views.ImageCreateView.as_view(success_url=reverse_lazy('dl:all')), name='image_create'),
    path('images/<int:pk>/update', views.ImageUpdateView.as_view(success_url=reverse_lazy('dl:all')), name='image_update'),
    path('images/<int:pk>/delete', views.ImageDeleteView.as_view(success_url=reverse_lazy('dl:all')), name='image_delete'),

    path('categories/', views.CategoryListView.as_view(), name='all_categories'),
    path('categories/<int:pk>', views.CategoryDetailView.as_view(), name='category_detail'),
    path('categories/create', views.CategoryCreateView.as_view(success_url=reverse_lazy('dl:all_categories')), name='category_create'),
    path('categories/<int:pk>/update', views.CategoryUpdateView.as_view(success_url=reverse_lazy('dl:all_categories')), name='category_update'),
    path('categories/<int:pk>/delete', views.CategoryDeleteView.as_view(success_url=reverse_lazy('dl:all_categories')), name='category_delete'),

    path('galleries/', views.GalleryListView.as_view(), name='all_galleries'),
    path('galleries/<int:pk>', views.GalleryDetailView.as_view(), name='gallery_detail'),
    path('galleries/create', views.GalleryCreateView.as_view(success_url=reverse_lazy('dl:all_galleries')), name='gallery_create'),
    path('galleries/<int:pk>/update', views.GalleryUpdateView.as_view(success_url=reverse_lazy('dl:all_galleries')), name='gallery_update'),
    path('galleries/<int:pk>/delete', views.GalleryDeleteView.as_view(success_url=reverse_lazy('dl:all_galleries')), name='gallery_delete'),

    path('stream_image/<int:pk>', views.stream_file, name='stream_image'),
]