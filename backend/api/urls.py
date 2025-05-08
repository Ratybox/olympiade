from django.urls import path
from . import views
from django.urls import include

urlpatterns = [
    path('upload/', views.upload, name='api-upload'),
    path('predict/', views.predict, name='api-predict'),
    path('predict-updrs/', views.predict_updrs, name='api-predict-updrs'),
    path('recordings/', views.recordings, name='api-recordings'),
    path('recordings/<str:recording_id>/', views.recording_detail, name='api-recording-detail'),
    path('parkinson/', views.parkinson, name='api-parkinson'),
    path('cough/', views.detect_cough, name='api-cough'),
    path('health/', views.health_check, name='api-health'),
]