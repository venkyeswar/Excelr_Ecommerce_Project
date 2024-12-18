from django.conf import settings
from django.urls import path,include
from django.conf.urls.static import static

urlpatterns = [
    # Your app URLs
    path('', include('prediction.urls')),  # Modify according to your app name
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS[0])
