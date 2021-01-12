from django.http import HttpResponse, JsonResponse
from django.urls import reverse_lazy, reverse
from django.views import View
from mld.models import ML_Type, Dataset, Algorithm, Entry, Graph, File_Type, Parameter_Name
from django.utils.decorators import method_decorator
from mld.owner import OwnerListView, OwnerDetailView, OwnerCreateView, OwnerUpdateView, OwnerDeleteView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.decorators.csrf import csrf_exempt
from django.views.generic import TemplateView
from django.db.models import Q
from mld.forms import CreateForm, CreateGraphForm, CreateDatasetForm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import seaborn as sns

from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import roc_curve, auc, classification_report, f1_score, precision_score, recall_score, precision_recall_curve, average_precision_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import mglearn

from io import StringIO, BytesIO
from django.shortcuts import render, redirect, get_object_or_404
from django.template import loader
import time

from django.core import serializers

class mld_form_js(View):
    def get(self, request):
        return render(request, 'mld/mld_form.js')


class return_algorithm_models(LoginRequiredMixin, View) :
    def get(self, request):
        data = serializers.serialize("json", Algorithm.objects.all(), fields=('title', 'ml_type', 'parameter_name', 'parameter_1_name', 'parameter_2_name', 'parameter_3_name', 'number_of_parameters'))
        return JsonResponse(data, safe=False)

class return_dataset_models(LoginRequiredMixin, View) :
    def get(self, request):
        if request.user.is_superuser == True :
            data = serializers.serialize("json", Dataset.objects.all(), fields=('title', 'ml_type'))
        else:
            data = serializers.serialize("json", Dataset.objects.all().filter(Q(public=True) | Q(owner=request.user)), fields=('title', 'ml_type'))
        return JsonResponse(data, safe=False)

class return_parameter_name_models(LoginRequiredMixin, View) :
    def get(self, request):
        data = serializers.serialize("json", Parameter_Name.objects.all(), fields=('title'))
        return JsonResponse(data, safe=False)

class return_entry_models(LoginRequiredMixin, View) :
    def get(self, request):
        if request.user.is_superuser == True :
            data = serializers.serialize("json", Entry.objects.all(), fields=('ml_type', 'dataset', 'algorithm'))
        else:
            data = serializers.serialize("json", Entry.objects.all().filter(owner=request.user), fields=('ml_type', 'dataset', 'algorithm'))
        return JsonResponse(data, safe=False)


# def return_models_json(request):
#     data = serializers.serialize("json", Algorithm.objects.all(), fields=('title', 'ml_type', 'parameter_name', 'parameter_1_name', 'parameter_2_name', 'parameter_3_name'))
#     return JsonResponse(data, safe=False)

# def return_algorithm_models(request):
#     data = serializers.serialize("json", Algorithm.objects.all(), fields=('title', 'ml_type', 'parameter_name', 'parameter_1_name', 'parameter_2_name', 'parameter_3_name', 'number_of_parameters'))
#     return JsonResponse(data, safe=False)

# def return_dataset_models(request):
#     data = serializers.serialize("json", Dataset.objects.all(), fields=('title', 'ml_type'))
#     return JsonResponse(data, safe=False)

# def return_parameter_name_models(request):
#     data = serializers.serialize("json", Parameter_Name.objects.all(), fields=('title'))
#     return JsonResponse(data, safe=False)

class MldHomeView(TemplateView):
    template_name = "mld/mld_home.html"

class MldAboutView(TemplateView):
    template_name = "mld/mld_about.html"

class MldListView(OwnerListView):
    model = Entry
    # By convention:
    template_name = "mld/mld_list.html"
    def get(self, request) :
        entry_list = Entry.objects.all()
        if request.user.is_superuser == True :
            ctx = {'entry_list' : entry_list}
        else:
            ctx = {'entry_list' : entry_list.filter(owner=request.user)}
        return render(request, self.template_name, ctx)

# def search(request):
#     status_list = Data.objects.all()
#     status_filter = status_list.filter(user=request.user) //get current user id
#     return render(request, 'users/data.html', {'filter': status_filter})

class GraphListView(OwnerListView):
    model = Graph
    # By convention:
    template_name = "mld/graph_list.html"
    def get(self, request) :
        graph_list = Graph.objects.all()
        if request.user.is_superuser == True :
            ctx = {'graph_list' : graph_list}
        else:
            ctx = {'graph_list' : graph_list.filter(owner=request.user)}
        return render(request, self.template_name, ctx)

class DatasetListView(OwnerListView):
    model = Dataset
    # By convention:
    template_name = "mld/dataset_list.html"
    def get(self, request) :
        dataset_list = Dataset.objects.all()
        if request.user.is_superuser == True :
            ctx = {'dataset_list' : dataset_list}
        else:
            ctx = {'dataset_list' : dataset_list.filter(Q(public=True) | Q(owner=request.user))}
        return render(request, self.template_name, ctx)

class AlgorithmListView(OwnerListView):
    model = Algorithm
    # By convention:
    template_name = "mld/algorithm_list.html"
    def get(self, request) :
        algorithm_list = Algorithm.objects.all()
        ctx = {'algorithm_list' : algorithm_list}
        return render(request, self.template_name, ctx)

class MldDetailView(OwnerDetailView):
    model = Entry
    template_name = "mld/mld_detail.html"
    def get(self, request, pk) :
        x = Entry.objects.get(id=pk)

        # if x.dataset.target_datatype == 'int':
        if x.dataset.target_datatype.title == 'int':
            target_data = np.frombuffer(x.target_data, dtype=np.uint64)
        # elif x.dataset.target_datatype == 'float':
        elif x.dataset.target_datatype.title == 'float':
            target_data = np.frombuffer(x.target_data)

        target_data_length = len(target_data)
        # roc_auc = x.roc_auc
        # tpr = np.frombuffer(x.tpr)
        # fpr = np.frombuffer(x.fpr)
        # thresholds = np.frombuffer(x.thresholds)

        context = { 'entry' : x, 'target_data' : target_data, 'target_data_length' : target_data_length}
        # context['plot'] = return_roc_graph(x.algorithm.title, tpr, fpr, thresholds, roc_auc)
        return render(request, self.template_name, context)



class GraphDetailView(OwnerDetailView):
    model = Graph
    template_name = "mld/graph_detail.html"
    def get(self, request, pk) :
        x = Graph.objects.get(id=pk)

        context = { 'graph' : x }

        return render(request, self.template_name, context)

def stream_file(request, pk):
    graph = get_object_or_404(Graph, id=pk)

    if graph.parameter_to_plot == 1:
        param_name = graph.algorithm.parameter_1_name.title
    elif graph.parameter_to_plot == 2:
        param_name = graph.algorithm.parameter_2_name.title
    elif graph.parameter_to_plot == 3:
        param_name = graph.algorithm.parameter_3_name.title

    response = HttpResponse()
    response['Content-Type'] = 'png'
    image = return_graph(graph.ml_type.id, graph.dataset.id, graph.algorithm.id, graph.parameter_to_plot, param_name)
    response['Content-Length'] = len(image)
    response.write(image)
    return response

def stream_roc_graph(request, pk):
    entry = get_object_or_404(Entry, id=pk)

    response = HttpResponse()
    response['Content-Type'] = 'png'

    tpr = np.frombuffer(entry.tpr)
    fpr = np.frombuffer(entry.fpr)
    thresholds = np.frombuffer(entry.thresholds)
    image = return_roc_graph(entry.algorithm.title, tpr, fpr, thresholds, entry.roc_auc)
    response['Content-Length'] = len(image)

    response.write(image)
    return response

def stream_precision_recall_graph(request, pk):
    entry = get_object_or_404(Entry, id=pk)

    response = HttpResponse()
    response['Content-Type'] = 'png'

    precision = np.frombuffer(entry.precision_c)
    recall = np.frombuffer(entry.recall_c)
    thresholds2 = np.frombuffer(entry.thresholds2)
    image = return_precision_recall_graph(entry.algorithm.title, precision, recall, thresholds2, entry.avg_precision)
    response['Content-Length'] = len(image)

    response.write(image)
    return response

def stream_heat_map(request, pk):
    entry = get_object_or_404(Entry, id=pk)

    response = HttpResponse()
    response['Content-Type'] = 'png'

    df = return_dataframe(entry.dataset)
    # target_data = np.frombuffer(entry.target_data)

    image = return_heat_map(df)
    response['Content-Length'] = len(image)
    response.write(image)
    return response

def stream_dist_plot(request, pk):
    entry = get_object_or_404(Entry, id=pk)

    response = HttpResponse()
    response['Content-Type'] = 'png'

    # if entry.dataset.target_datatype == 'int':
    if entry.dataset.target_datatype.title == 'int':
        target_data = np.frombuffer(entry.target_data, dtype=np.int64)
    # elif entry.dataset.target_datatype == 'float':
    elif entry.dataset.target_datatype.title == 'float':
        target_data = np.frombuffer(entry.target_data)

    # target_data = np.frombuffer(entry.target_data)

    image = return_dist_plot(target_data)
    response['Content-Length'] = len(image)
    response.write(image)
    return response

def return_dist_plot(target_data):

    # fig = plt.figure()
    # sns.set(rc={'figure.figsize':(11.7,8.27)})
    plt.figure()
    sns.displot(target_data, bins=15, kde=True, height=3)

    # sns.distplot(target_data, bins=30)

    # fig = sns.displot(target_data, bins=30)
    # plt.show()

    imgdata = BytesIO()
    # fig2 = fig.get_figure()
    # fig2.savefig(imgdata, format='png', dpi=300)
    plt.savefig(imgdata, format='png', dpi=300)
    # fig.savefig(imgdata, format='png', dpi=300)
    plt.clf()
    imgdata.seek(0)

    data = imgdata.getvalue()
    return data

def return_heat_map(df):

    plt.figure(figsize=(16,16))
    correlation_matrix = df.corr().round(2)
    # sns.heatmap(data=correlation_matrix, annot=True, annot_kws = {'size':'small'} )
    sns.heatmap(data=correlation_matrix, annot=True )

    imgdata = BytesIO()
    plt.savefig(imgdata, format='png', dpi=300)
    plt.clf()
    imgdata.seek(0)

    data = imgdata.getvalue()
    return data

def return_roc_graph(algorithm_title, tpr, fpr, thresholds, roc_auc):
    fig = plt.figure()
    sns.set_theme()
    # sns.set_style("ticks")

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    if algorithm_title == 'Logistic Regression' or algorithm_title == 'Linear Support Vector Classifier':
        threshold_default = np.argmin(np.abs(thresholds))
        threshold_label = "Threshold Zero"
    else:
        threshold_default = np.argmin(np.abs(thresholds - 0.5))
        threshold_label = "Threshold = 0.5"

    plt.plot(fpr[threshold_default], tpr[threshold_default], 'o', markersize=10, label=threshold_label, fillstyle='none', c='k', mew=2)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt_title = 'ROC Curve of ' + algorithm_title
    plt.title(plt_title)

    # fig.tight_layout()
    plt.show()

    # imgdata = StringIO()
    # fig.savefig(imgdata, format='svg')
    imgdata = BytesIO()
    fig.savefig(imgdata, format='png', dpi=300)
    plt.clf()
    imgdata.seek(0)

    data = imgdata.getvalue()
    return data

def return_precision_recall_graph(algorithm_title, precision, recall, thresholds, avg_precision):
    fig = plt.figure()
    sns.set_theme()

    plt.plot(precision, recall, 'b', label = 'Avg. Precision = %0.2f' % avg_precision)

    if algorithm_title == 'Logistic Regression' or algorithm_title == 'Linear Support Vector Classifier':
        threshold_default = np.argmin(np.abs(thresholds))
        threshold_label = "Threshold Zero"
    else:
        threshold_default = np.argmin(np.abs(thresholds - 0.5))
        threshold_label = "Threshold = 0.5"

    plt.plot(precision[threshold_default], recall[threshold_default], 'o', markersize=10, label=threshold_label, fillstyle='none', c='k', mew=2)
    plt.legend(loc='best')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt_title = 'Precision-Recall Curve of ' + algorithm_title
    plt.title(plt_title)

    # fig.tight_layout()
    plt.show()

    # imgdata = StringIO()
    # fig.savefig(imgdata, format='svg')
    imgdata = BytesIO()
    fig.savefig(imgdata, format='png', dpi=300)
    plt.clf()
    imgdata.seek(0)

    data = imgdata.getvalue()
    return data

def return_graph(ml_type_index, dataset_index, algorithm_index, parameter_number, parameter_name):

    if parameter_number == 1:
        entries = Entry.objects.filter(algorithm_id=algorithm_index, dataset_id=dataset_index).order_by('-parameter_1_value').values()
    elif parameter_number == 2:
        entries = Entry.objects.filter(algorithm_id=algorithm_index, dataset_id=dataset_index).order_by('-parameter_2_value').values()
    elif parameter_number == 3:
        entries = Entry.objects.filter(algorithm_id=algorithm_index, dataset_id=dataset_index).order_by('-parameter_3_value').values()

    training_accuracy = []
    test_accuracy = []
    r2_train_score = []
    r2_test_score = []
    parameter_values = []

    for item in entries:
        if parameter_number == 1:
            parameter_values.append(item['parameter_1_value'])
        elif parameter_number == 2:
            parameter_values.append(item['parameter_2_value'])
        elif parameter_number == 3:
            parameter_values.append(item['parameter_3_value'])

        if ml_type_index == 1:
            training_accuracy.append(item['train_accuracy'])
            test_accuracy.append(item['test_accuracy'])
        else:
            r2_train_score.append(item['r2_train_score'])
            r2_test_score.append(item['r2_test_score'])

    sns.set_theme()
    # plt.style.use('seaborn-darkgrid')
    fig = plt.figure()

    if ml_type_index == 1:
        plt.plot(parameter_values, training_accuracy, label="training accuracy", marker='.', markersize=6)
        plt.plot(parameter_values, test_accuracy, label="test accuracy", linestyle='dashed', marker='.', markersize=6)
        plt.ylabel("Accuracy")
    else:
        plt.plot(parameter_values, r2_train_score, label="R-squared train score", marker='.', markersize=6)
        plt.plot(parameter_values, r2_test_score, label="R-squared test score", linestyle='dashed',  marker='.', markersize=6)
        plt.ylabel("R-squared score")

    plt.xlabel(parameter_name)
    plt.legend()

    # imgdata = StringIO()
    # fig.savefig(imgdata, format='svg')

    imgdata = BytesIO()
    fig.savefig(imgdata, format='png', dpi=300)
    plt.clf()
    imgdata.seek(0)

    data = imgdata.getvalue()
    return data


class DatasetDetailView(OwnerDetailView):
    model = Dataset
    template_name = "mld/dataset_detail.html"
    def get(self, request, pk) :
        x = Dataset.objects.get(id=pk)

        context = { 'dataset' : x }

        df = return_dataframe(x)

        ###
        # df_new = return_dataframe(obj.dataset)
        # s = df.iloc[:, -1]
        # if x.target_data != None:
        #     target_data = np.frombuffer(x.target_data, dtype=np.int64)

        html_data = df.to_html(classes='table table-responsive table-bordered table-hover table-condensed table-striped')

        context['html_data'] = html_data
        # context['target_data'] = target_data
        return render(request, self.template_name, context)

class AlgorithmDetailView(OwnerDetailView):
    model = Algorithm
    template_name = "mld/algorithm_detail.html"
    def get(self, request, pk) :
        x = Algorithm.objects.get(id=pk)

        context = { 'algorithm' : x }
        return render(request, self.template_name, context)

class MldCreateView(LoginRequiredMixin, View):
    template_name = 'mld/mld_form.html'
    success_url = reverse_lazy('mld:all')

    def get(self, request, pk=None):
        form = CreateForm()
        ctx = {'form': form}
        return render(request, self.template_name, ctx)

    def post(self, request, pk=None):
        form = CreateForm(request.POST, request.FILES or None)

        if not form.is_valid():
            ctx = {'form': form}
            return render(request, self.template_name, ctx)

        # Add owner to the model before saving
        entry = form.save(commit=False)
        entry.owner = self.request.user
        entry = ml_train_test(entry)
        entry.save()
        return redirect(self.success_url)

class GraphCreateView(LoginRequiredMixin, View):
    template_name = 'mld/graph_form.html'
    success_url = reverse_lazy('mld:all')

    def get(self, request, pk=None):
        form = CreateGraphForm()
        ctx = {'form': form}
        return render(request, self.template_name, ctx)

    def post(self, request, pk=None):
        form = CreateGraphForm(request.POST, request.FILES or None)

        if not form.is_valid():
            ctx = {'form': form}
            return render(request, self.template_name, ctx)

        # Add owner to the model before saving
        graph = form.save(commit=False)
        graph.owner = self.request.user

        graph.save()
        return redirect(self.success_url)

class DatasetCreateView(LoginRequiredMixin, View):
    template_name = 'mld/dataset_form.html'
    success_url = reverse_lazy('mld:all_datasets')

    def get(self, request, pk=None):
        form = CreateDatasetForm()
        ctx = {'form': form}
        return render(request, self.template_name, ctx)

    def post(self, request, pk=None):
        form = CreateDatasetForm(request.POST, request.FILES or None)

        if not form.is_valid():
            ctx = {'form': form}
            return render(request, self.template_name, ctx)

        # Add owner to the model before saving
        dataset = form.save(commit=False)
        dataset.owner = self.request.user

        dataset = load_dataset(dataset)

        dataset.save()

        return redirect(self.success_url)

def return_dataframe(dataset):
    if dataset.file_type.title == 'CSV':
        if dataset.has_header == True:
            df = pd.read_csv(BytesIO(dataset.data), header=0)
        else:
            df = pd.read_csv(BytesIO(dataset.data), header=None)
    elif dataset.file_type.title == 'Excel .xlsx':
        if dataset.has_header == True:
            df = pd.read_excel(BytesIO(dataset.data), engine='openpyxl', skiprows=[0], header=0)
        else:
            df = pd.read_excel(BytesIO(dataset.data), engine='openpyxl', skiprows=[0], header=None)
    elif dataset.file_type.title == 'Excel .xls':
        if dataset.has_header == True:
            df = pd.read_excel(BytesIO(dataset.data), skiprows=[0], header=0)
        else:
            df = pd.read_excel(BytesIO(dataset.data), skiprows=[0], header=None)
    return df

def load_dataset(dataset):
    if dataset.file_type.title == 'CSV':
        if dataset.has_header == True:
            df = pd.read_csv(BytesIO(dataset.data), header=0)
        else:
            df = pd.read_csv(BytesIO(dataset.data), header=None)
        df_data = df.iloc[:,:-1]
        df_target = df.iloc[:, -1]
    elif dataset.file_type.title == 'Excel .xlsx':
        if dataset.has_header == True:
            df = pd.read_excel(BytesIO(dataset.data), engine='openpyxl', skiprows=[0], header=0)
        else:
            df = pd.read_excel(BytesIO(dataset.data), engine='openpyxl', skiprows=[0], header=None)
        df_data = df.iloc[:,:-1]
        df_target = df.iloc[:, -1]
        df_target=df_target.astype('str')
    elif dataset.file_type.title == 'Excel .xls':
        if dataset.has_header == True:
            df = pd.read_excel(BytesIO(dataset.data), skiprows=[0], header=0)
        else:
            df = pd.read_excel(BytesIO(dataset.data), skiprows=[0], header=None)
        df_data = df.iloc[:,:-1]
        df_target = df.iloc[:, -1]
        df_target=df_target.astype('str')

    dataset.dimensionality = len(df_data.columns)
    dataset.samples = len(df)
    df_target_float = df.iloc[:, -1].astype(float)
    s = df_target_float.to_numpy()
    dataset.target_data = s.tobytes()
    if dataset.ml_type.id == 1:
        vcounts = df_target.value_counts()
        dataset.number_of_classes = len(vcounts)
        dataset.class_1_count = vcounts.iloc[0]
        dataset.class_2_count = vcounts.iloc[1]
        dataset.class_1_label = str(vcounts.index.values[0])
        dataset.class_2_label = str(vcounts.index.values[1])
    else:
        dataset.target_mean = '{:0.2f}'.format(df_target.mean())
        dataset.target_std = '{:0.2f}'.format(df_target.std())
        dataset.target_min = '{:0.2f}'.format(df_target.min())
        dataset.target_max = '{:0.2f}'.format(df_target.max())
    return dataset

class MldUpdateView(LoginRequiredMixin, View):
    template_name = 'mld/mld_form.html'
    success_url = reverse_lazy('mld:all')

    def get(self, request, pk):
        entry = get_object_or_404(Entry, id=pk, owner=self.request.user)
        form = CreateForm(instance=entry)
        ctx = {'form': form}
        return render(request, self.template_name, ctx)

    def post(self, request, pk=None):
        entry = get_object_or_404(Entry, id=pk, owner=self.request.user)
        form = CreateForm(request.POST, request.FILES or None, instance=entry)

        if not form.is_valid():
            ctx = {'form': form}
            return render(request, self.template_name, ctx)

        entry = form.save(commit=False)
        entry = ml_train_test(entry)
        entry.save()

        return redirect(self.success_url)

class GraphUpdateView(LoginRequiredMixin, View):
    template_name = 'mld/graph_form.html'
    success_url = reverse_lazy('mld:all_graphs')

    def get(self, request, pk):
        graph = get_object_or_404(Graph, id=pk, owner=self.request.user)
        form = CreateGraphForm(instance=graph)
        ctx = {'form': form}
        return render(request, self.template_name, ctx)

    def post(self, request, pk=None):
        graph = get_object_or_404(Graph, id=pk, owner=self.request.user)
        form = CreateGraphForm(request.POST, request.FILES or None, instance=graph)

        if not form.is_valid():
            ctx = {'form': form}
            return render(request, self.template_name, ctx)

        graph = form.save(commit=False)


        graph.save()

        return redirect(self.success_url)

class DatasetUpdateView(LoginRequiredMixin, View):
    template_name = 'mld/dataset_form.html'
    success_url = reverse_lazy('mld:all_datasets')

    def get(self, request, pk):
        dataset = get_object_or_404(Dataset, id=pk, owner=self.request.user)
        form = CreateDatasetForm(instance=dataset)
        ctx = {'form': form}
        return render(request, self.template_name, ctx)

    def post(self, request, pk=None):
        dataset = get_object_or_404(Dataset, id=pk, owner=self.request.user)
        form = CreateDatasetForm(request.POST, request.FILES or None, instance=dataset)

        if not form.is_valid():
            ctx = {'form': form}
            return render(request, self.template_name, ctx)

        dataset = form.save(commit=False)

        dataset = load_dataset(dataset)

        dataset.save()

        return redirect(self.success_url)


class MldDeleteView(OwnerDeleteView):
    model = Entry

class GraphDeleteView(OwnerDeleteView):
    model = Graph

class DatasetDeleteView(OwnerDeleteView):
    model = Dataset

def ml_train_test(obj):

    start = time.time()

    if obj.dataset.file_type.title == 'CSV':
        if obj.dataset.has_header == True:
            df = pd.read_csv(BytesIO(obj.dataset.data), header=0)
        else:
            df = pd.read_csv(BytesIO(obj.dataset.data), header=None)
        df_data = df.iloc[:,:-1]
        df_target = df.iloc[:, -1]
    elif obj.dataset.file_type.title == 'Excel .xlsx':
        if obj.dataset.has_header == True:
            df = pd.read_excel(BytesIO(obj.dataset.data), engine='openpyxl', skiprows=[0], header=0)
        else:
            df = pd.read_excel(BytesIO(obj.dataset.data), engine='openpyxl', skiprows=[0], header=None)
        df_data = df.iloc[:,:-1]
        df_target = df.iloc[:, -1]
        # df_target=df_target.astype('str')
    elif obj.dataset.file_type.title == 'Excel .xls':
        if obj.dataset.has_header == True:
            df = pd.read_excel(BytesIO(obj.dataset.data), skiprows=[0], header=0)
        else:
            df = pd.read_excel(BytesIO(obj.dataset.data), skiprows=[0], header=None)
        df_data = df.iloc[:,:-1]
        df_target = df.iloc[:, -1]
        # df_target=df_target.astype('str')

    # train_test split for binary classification or regression dataset
    if obj.dataset.ml_type.id == 1:
        X_train, X_test, y_train, y_test = train_test_split(df_data, df_target, stratify=df_target, random_state=0)
    elif obj.dataset.ml_type.id == 2:
        X_train, X_test, y_train, y_test = train_test_split(df_data, df_target, test_size=0.2, random_state=5)

    model = obj.algorithm.index

    param1 = obj.parameter_1_value
    # try:
    #     param1 = float(param1)
    # except:
    #     param1 = 1

    param2 = obj.parameter_2_value
    # try:
    #     param2 = float(param2)
    # except:
    #     param2 = 1

    param3 = obj.parameter_3_value
    # try:
    #     param3 = float(param3)
    # except:
    #     param3 = 1

    if model == 1:
        clf = KNeighborsClassifier(n_neighbors=int(param1)).fit(X_train, y_train)
    elif model == 2:
        clf = LogisticRegression(C=param1).fit(X_train, y_train)
    elif model == 3:
        clf = LinearSVC(C=param1).fit(X_train, y_train)
    elif model == 4:
        clf = DecisionTreeClassifier(max_depth = int(param1), random_state=0).fit(X_train, y_train)
    elif model == 5:
        clf = RandomForestClassifier(n_estimators = int(param1), random_state=0).fit(X_train, y_train)
    elif model == 6:
        clf = GradientBoostingClassifier(max_depth = int(param1), random_state=0).fit(X_train, y_train)
    elif model == 7:
        clf = MLPClassifier(solver='lbfgs',  hidden_layer_sizes=[int(param1)], max_iter=int(param2), alpha=param3, random_state=0).fit(X_train, y_train)
    elif model == 8:
        clf = GaussianNB(var_smoothing = param1).fit(X_train, y_train)
    elif model == 9:
        clf = MultinomialNB(alpha = param1).fit(X_train, y_train)
    elif model == 10:
        reg = KNeighborsRegressor(n_neighbors=int(param1)).fit(X_train, y_train)
    elif model == 11:
        reg = LinearRegression().fit(X_train, y_train)
    elif model == 12:
        reg = Ridge(alpha=param1).fit(X_train, y_train)
    elif model == 13:
        reg = Lasso(alpha=param1).fit(X_train, y_train)


    end = time.time()

    modeltime = end - start
    modeltime = '{:0.3f}'.format(modeltime)
    obj.time = modeltime

    # if model is binary classification do those metrics
    if model < 10:
        train_accuracy = clf.score(X_train, y_train)
        train_accuracy = '{:0.3f}'.format(train_accuracy)
        y_pred = clf.predict(X_test)
        precision = precision_score(y_test, y_pred, average="macro")
        precision = '{:0.3f}'.format(precision)
        f1 = f1_score(y_test, y_pred, average="macro")
        f1 = '{:0.3f}'.format(f1)
        recall = recall_score(y_test, y_pred, average="macro")
        recall = '{:0.3f}'.format(recall)
        test_accuracy = clf.score(X_test, y_test)
        test_accuracy = '{:0.3f}'.format(test_accuracy)

        obj.precision = precision
        obj.recall = recall
        obj.f1 = f1
        obj.train_accuracy = train_accuracy
        obj.test_accuracy = test_accuracy

        confusion = confusion_matrix(y_test, y_pred)
        obj.TN  = confusion[0,0]
        obj.FP = confusion[0,1]
        obj.FN = confusion[1,0]
        obj.TP = confusion[1,1]

        if model == 2 or model == 3:
            # if model is Logistic Regression or LinearSVC use decision_function to calculate ROC curve:
            fpr, tpr, thresholds = roc_curve(y_test, clf.decision_function(X_test))
            precision_c, recall_c, thresholds2 = precision_recall_curve(y_test, clf.decision_function(X_test))
            avg_precision = average_precision_score(y_test, clf.decision_function(X_test))
        else:
            # otherwise use predict_proba to calculate ROC curve:
            y_scores = clf.predict_proba(X_test)
            fpr, tpr, thresholds = roc_curve(y_test, y_scores[:, 1])
            precision_c, recall_c, thresholds2 = precision_recall_curve(y_test, clf.predict_proba(X_test)[:,1])
            avg_precision = average_precision_score(y_test, clf.predict_proba(X_test)[:,1])

        avg_precision = '{:0.3f}'.format(avg_precision)
        obj.avg_precision = avg_precision

        roc_auc = auc(fpr, tpr)
        roc_auc = '{:0.3f}'.format(roc_auc)
        obj.roc_auc = roc_auc

        # save numpy arrays for ROC curve to binary fields of entry model
        obj.fpr = fpr.tobytes()
        obj.tpr = tpr.tobytes()
        obj.thresholds = thresholds.tobytes()

        obj.precision_c = precision_c.tobytes()
        obj.recall_c = recall_c.tobytes()
        obj.thresholds2 = thresholds2.tobytes()
        df_new = return_dataframe(obj.dataset)
        s = df_new.iloc[:, -1]
        target_data = s.to_numpy()
        obj.target_data = target_data.tobytes()
    else: # if model is regression do regression metrics
        y_train_predict = reg.predict(X_train)
        y_test_predict = reg.predict(X_test)

        rmse_train_score = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
        rmse_train_score = '{:0.3f}'.format(rmse_train_score)
        rmse_test_score = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
        rmse_test_score = '{:0.3f}'.format(rmse_test_score)

        r2_train_score = r2_score(y_train, y_train_predict)
        r2_train_score = '{:0.3f}'.format(r2_train_score)

        r2_test_score = r2_score(y_test, y_test_predict)
        r2_test_score = '{:0.3f}'.format(r2_test_score)

        obj.r2_train_score = r2_train_score
        obj.r2_test_score = r2_test_score
        obj.rmse_train = rmse_train_score
        obj.rmse_test = rmse_test_score

        df_new = return_dataframe(obj.dataset)
        s = df_new.iloc[:, -1]
        target_data = s.to_numpy()
        obj.target_data = target_data.tobytes()

    return obj