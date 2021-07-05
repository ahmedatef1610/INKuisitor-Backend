from django.contrib import admin
from django.urls import path, include
from clerk import views

urlpatterns = [
    path('create', views.createprofile_view, name="createprofile"),
    path('verify', views.verify_view, name="verifyclient"),
    path('details', views.clientdetails_view, name="clientdetails"),
]
