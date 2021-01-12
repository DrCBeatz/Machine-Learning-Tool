from django.http import HttpResponse
from django.urls import reverse_lazy
from django.views import View
from dl.models import Gallery, Category, Image
from dl.owner import OwnerListView, OwnerDetailView, OwnerCreateView, OwnerUpdateView, OwnerDeleteView

from django.shortcuts import render, redirect, get_object_or_404
from django.views.generic import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin

from dl.forms import CreateImageForm, CreateCategoryForm, CreateGalleryForm


class DlAboutView(TemplateView):
    template_name = "dl/dl_about.html"
    def get_context_data(self, **kwargs):
        context = super(DlAboutView, self).get_context_data(**kwargs)
        gallery_list = Gallery.objects.all()
        category_list = Category.objects.all()
        context = {'gallery_list': gallery_list, 'category_list': category_list}
        return context

class ImageListView(OwnerListView):
    model = Image
    # By convention:
    template_name = "dl/image_list.html"
    def get(self, request) :
        # image_list = Image.objects.all()
        image_list = Image.objects.all().order_by('-pk')
        category_list = Category.objects.all()
        gallery_list = Gallery.objects.all()
        ctx = {'image_list' : image_list, 'gallery_list' : gallery_list, 'category_list' : category_list}
        return render(request, self.template_name, ctx)


class CategoryListView(OwnerListView):
    model = Category
    # By convention:
    template_name = "dl/category_list.html"
    def get(self, request) :
        category_list = Category.objects.all()
        gallery_list = Gallery.objects.all()

        ctx = {'category_list' :category_list, 'gallery_list':gallery_list}
        return render(request, self.template_name, ctx)

class GalleryListView(OwnerListView):
    model = Gallery
    # By convention:
    template_name = "dl/gallery_list.html"
    def get(self, request) :
        gallery_list = Gallery.objects.all()
        category_list = Category.objects.all()
        ctx = {'gallery_list' :gallery_list, 'category_list':category_list}
        return render(request, self.template_name, ctx)

class ImageDetailView(OwnerDetailView):
    model = Image
    template_name = "dl/image_detail.html"
    def get(self, request, pk) :
        image = Image.objects.get(id=pk)
        gallery_list = Gallery.objects.all()
        category_list = Category.objects.all()
        context = { 'image' : image, 'gallery_list': gallery_list, 'category_list':category_list }
        return render(request, self.template_name, context)

class CategoryDetailView(OwnerDetailView):
    model = Category
    template_name = "dl/category_detail.html"
    def get(self, request, pk) :
        category = Category.objects.get(id=pk)
        image_list = Image.objects.all().filter(category=category.id)
        gallery_list = Gallery.objects.all()
        category_list = Category.objects.all()
        context = { 'category' : category, 'image_list' : image_list,  'gallery_list': gallery_list, 'category_list':category_list }
        return render(request, self.template_name, context)

class GalleryDetailView(OwnerDetailView):
    model = Gallery
    template_name = "dl/gallery_detail.html"
    def get(self, request, pk) :
        gallery = Gallery.objects.get(id=pk)
        category_list = Category.objects.all().filter(gallery=gallery.id)
        gallery_list = Gallery.objects.all()
        context = { 'gallery' : gallery, 'category_list':category_list, 'gallery_list': gallery_list}
        return render(request, self.template_name, context)

class ImageCreateView(LoginRequiredMixin, View):
    template_name = 'dl/image_form.html'
    success_url = reverse_lazy('dl:all')

    def get(self, request, pk=None):
        form = CreateImageForm()
        ctx = {'form': form}
        return render(request, self.template_name, ctx)

    def post(self, request, pk=None):
        form = CreateImageForm(request.POST, request.FILES or None)

        if not form.is_valid():
            ctx = {'form': form}
            return render(request, self.template_name, ctx)

        # Add owner to the model before saving
        image = form.save(commit=False)
        image.owner = self.request.user

        image.save()
        return redirect(self.success_url)

class CategoryCreateView(LoginRequiredMixin, View):
    template_name = 'dl/category_form.html'
    success_url = reverse_lazy('dl:all_categories')

    def get(self, request, pk=None):
        form = CreateCategoryForm()
        ctx = {'form': form}
        return render(request, self.template_name, ctx)

    def post(self, request, pk=None):
        form = CreateCategoryForm(request.POST, request.FILES or None)
        if not form.is_valid():
            ctx = {'form': form}
            return render(request, self.template_name, ctx)

        # Add owner to the model before saving
        category = form.save(commit=False)
        category.owner = self.request.user

        category.save()
        return redirect(self.success_url)

class GalleryCreateView(LoginRequiredMixin, View):
    template_name = 'dl/gallery_form.html'
    success_url = reverse_lazy('dl:all_galleries')

    def get(self, request, pk=None):
        form = CreateGalleryForm()
        ctx = {'form': form}
        return render(request, self.template_name, ctx)

    def post(self, request, pk=None):
        form = CreateGalleryForm(request.POST, request.FILES or None)
        if not form.is_valid():
            ctx = {'form': form}
            return render(request, self.template_name, ctx)

        # Add owner to the model before saving
        gallery = form.save(commit=False)
        gallery.owner = self.request.user

        gallery.save()
        return redirect(self.success_url)

class ImageUpdateView(LoginRequiredMixin, View):
    template_name = 'dl/image_form.html'
    success_url = reverse_lazy('dl:all')

    def get(self, request, pk):
        image = get_object_or_404(Image, id=pk, owner=self.request.user)
        form = CreateImageForm(instance=image)
        ctx = {'form': form}
        return render(request, self.template_name, ctx)

    def post(self, request, pk=None):
        image = get_object_or_404(Image, id=pk, owner=self.request.user)
        form = CreateImageForm(request.POST, request.FILES or None, instance=image)

        if not form.is_valid():
            ctx = {'form': form}
            return render(request, self.template_name, ctx)

        image = form.save(commit=False)
        image.save()

        return redirect(self.success_url)


class CategoryUpdateView(LoginRequiredMixin, View):
    template_name = 'dl/category_form.html'
    success_url = reverse_lazy('dl:all_categories')

    def get(self, request, pk):
        category = get_object_or_404(Category, id=pk, owner=self.request.user)
        form = CreateCategoryForm(instance=category)
        ctx = {'form': form}
        return render(request, self.template_name, ctx)

    def post(self, request, pk=None):
        category = get_object_or_404(Category, id=pk, owner=self.request.user)
        form = CreateCategoryForm(request.POST, request.FILES or None, instance=category)

        if not form.is_valid():
            ctx = {'form': form}
            return render(request, self.template_name, ctx)

        category = form.save(commit=False)
        category.save()

        return redirect(self.success_url)

class GalleryUpdateView(LoginRequiredMixin, View):
    template_name = 'dl/gallery_form.html'
    success_url = reverse_lazy('dl:all_galleries')

    def get(self, request, pk):
        gallery = get_object_or_404(Gallery, id=pk, owner=self.request.user)
        form = CreateGalleryForm(instance=gallery)
        ctx = {'form': form}
        return render(request, self.template_name, ctx)

    def post(self, request, pk=None):
        gallery = get_object_or_404(Gallery, id=pk, owner=self.request.user)
        form = CreateGalleryForm(request.POST, request.FILES or None, instance=gallery)

        if not form.is_valid():
            ctx = {'form': form}
            return render(request, self.template_name, ctx)

        gallery = form.save(commit=False)
        gallery.save()

        return redirect(self.success_url)

class ImageDeleteView(OwnerDeleteView):
    model = Image

class CategoryDeleteView(OwnerDeleteView):
    model = Category

class GalleryDeleteView(OwnerDeleteView):
    model = Gallery

def stream_file(request, pk):
    image = get_object_or_404(Image, id=pk)
    response = HttpResponse()

    response['Content-Type'] = image.content_type

    response['Content-Length'] = len(image.image)
    response.write(image.image)
    return response




