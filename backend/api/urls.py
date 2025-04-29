from django.urls import path
from . import views
from django.urls import include

urlpatterns = [
    path('upload/', views.upload, name='upload'),
    path('predict/', views.predict, name='predict'),
    path('recordings/', views.recordings, name='recordings'),
    path('recordings/<int:recording_id>/', views.recording_detail, name='recording_detail'),
]