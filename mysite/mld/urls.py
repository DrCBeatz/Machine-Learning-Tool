from django.urls import path, reverse_lazy
from . import views
from django.views.generic import TemplateView

app_name = 'mld'
urlpatterns = [
# path('', views.index, name='index'),
    # path('', views.MldListView.as_view()),
    path('', views.MldHomeView.as_view(), name='mld_home'),
    path('about/', views.MldAboutView.as_view(), name='mld_about'),
    path('mld', views.MldListView.as_view(), name='all'),
    path('mld/<int:pk>', views.MldDetailView.as_view(), name='mld_detail'),
    path('mld/create', views.MldCreateView.as_view(success_url=reverse_lazy('mld:all')), name='mld_create'),
    path('mld/<int:pk>/update', views.MldUpdateView.as_view(success_url=reverse_lazy('mld:all')), name='mld_update'),
    path('mld/<int:pk>/delete', views.MldDeleteView.as_view(success_url=reverse_lazy('mld:all')), name='mld_delete'),

    # path('graphs/', views.GraphListView.as_view()),
    path('graphs/', views.GraphListView.as_view(), name='all_graphs'),
    path('graphs/<int:pk>', views.GraphDetailView.as_view(), name='graph_detail'),
    path('graphs/create', views.GraphCreateView.as_view(success_url=reverse_lazy('mld:all_graphs')), name='graph_create'),
    path('graphs/<int:pk>/update', views.GraphUpdateView.as_view(success_url=reverse_lazy('mld:all_graphs')), name='graph_update'),
    path('graphs/<int:pk>/delete', views.GraphDeleteView.as_view(success_url=reverse_lazy('mld:all_graphs')), name='graph_delete'),


    path('datasets/', views.DatasetListView.as_view(), name='all_datasets'),
    path('datasets/<int:pk>', views.DatasetDetailView.as_view(), name='dataset_detail'),
    path('datasets/create', views.DatasetCreateView.as_view(success_url=reverse_lazy('mld:all_datasets')), name='dataset_create'),
    path('datasets/<int:pk>/update', views.DatasetUpdateView.as_view(success_url=reverse_lazy('mld:all_datasets')), name='dataset_update'),
    path('datasets/<int:pk>/delete', views.DatasetDeleteView.as_view(success_url=reverse_lazy('mld:all_datasets')), name='dataset_delete'),

    path('algorithms/', views.AlgorithmListView.as_view(), name='all_algorithms'),
    path('algorithms/<int:pk>', views.AlgorithmDetailView.as_view(), name='algorithm_detail'),

    path('graph_picture/<int:pk>', views.stream_file, name='graph_picture'),
    path('roc_graph_picture/<int:pk>', views.stream_roc_graph, name='roc_graph_picture'),
    path('precision_recall_graph_picture/<int:pk>', views.stream_precision_recall_graph, name='precision_recall_graph_picture'),
    path('dist_plot_picture/<int:pk>', views.stream_dist_plot, name='dist_plot_picture'),
    path('heat_map_picture/<int:pk>', views.stream_heat_map, name='heat_map_picture'),


    path('return_algorithm_models', views.return_algorithm_models.as_view(), name='return_algorithm_models'),
    path('return_dataset_models', views.return_dataset_models.as_view(), name='return_dataset_models'),
    path('return_parameter_name_models', views.return_parameter_name_models.as_view(), name='return_parameter_name_models'),

    # old def views for returning models as json:
    # path('return_models_json/', views.return_models_json, name='return_models_json'),
    # path('return_algorithm_models/', views.return_algorithm_models, name='return_algorithm_models'),
    # path('return_dataset_models/', views.return_dataset_models, name='return_dataset_models'),
    # path('return_parameter_name_models/', views.return_parameter_name_models, name='return_parameter_name_models'),

    # url(r'^mld_form_js', TemplateView.as_view(template_name='home/drcbeatz/django_projects/mysite/mld/js/mld_form.js'), name='mld_form_js'),
    # path('mld_form_js', TemplateView.as_view(template_name='home/drcbeatz/django_projects/mysite/mld/js/mld_form.js'), name='mld_form_js'),
    # path('mld_form_js', TemplateView.as_view(template_name='home/drcbeatz/django_projects/mysite/mld/js/mld_form.js'), name='mld_form_js'),
    # path('mld_form_js', views.mld_form_js.as_view()),
    path('mld_form_js', TemplateView.as_view(template_name='mld/mld_form.js')),
 ]