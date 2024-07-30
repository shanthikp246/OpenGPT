kubectl apply -f k8s/deployment-rabbitmq.yaml
kubectl apply -f k8s/deployment-pushgateway.yaml

kubectl apply -f k8s/deployment-fastapi.yaml
kubectl apply -f k8s/service-fastapi.yaml

kubectl apply -f k8s/deployment-celery.yaml
kubectl apply -f k8s/service-celery.yaml

kubectl apply -f k8s/hpa-celery.yaml
kubectl apply -f k8s/hpa-fastapi.yaml

