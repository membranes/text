<br>

### Notes

Important set up requirements for [main.yml](main.yml):

```yaml
with:
  role-to-assume: arn:aws:iam::${{ secrets.AWS_ENTRY }}:role/${{ secrets.AWS_ARN_ECR_ACTIONS }}
  aws-region: ${{ secrets.AWS_REGION }}
```

and

```shell
docker build . --file Dockerfile --tag "$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG"
```

whereby

```yaml
env:
  ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
  ECR_REPOSITORY: ${{ secrets.ECR_REPOSITORY }}
  IMAGE_TAG: ${{ github.event.repository.name }}
```

<br>
<br>

<br>
<br>

<br>
<br>

<br>
<br>
