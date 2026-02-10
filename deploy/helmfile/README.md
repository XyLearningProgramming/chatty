## Deploy with Helmfile

### Required GitHub Secrets

These secrets must be configured in the GitHub repository under **Settings > Environments > prod**:

| Secret | Description | Example |
|--------|-------------|---------|
| `REGISTRY_HOST` | Container registry host | `ghcr.io` |
| `REGISTRY_USERNAME` | Registry login username | `XyLearningProgramming` |
| `REGISTRY_PASSWORD` | Registry login password or token | (GitHub PAT with `write:packages`) |
| `IMAGE_NAME` | Full image repository name | `ghcr.io/xylearningprogramming/chatty` |
| `KUBE_CONFIG_DATA` | Base64-encoded kubeconfig for the target cluster | `cat ~/.kube/config \| base64` |
| `CHATTY_ENV` | Contents of `chatty.env` secret file (see `deploy/secret/chatty.env.example`) | LLM endpoint, API key, etc. |

### Local Development

1. Copy the example env file and fill in real values:

```bash
cp deploy/secret/chatty.env.example deploy/secret/chatty.env
```

2. Apply the secret to your local cluster:

```bash
./deploy/secret/apply.sh
```

3. Deploy with helmfile (default environment):

```bash
cd deploy/helmfile
helmfile apply
```

4. Deploy with helmfile (prod environment):

```bash
cd deploy/helmfile
IMAGE_REPOSITORY=ghcr.io/xylearningprogramming/chatty IMAGE_TAG=latest helmfile apply --environment prod
```

### Helpers

1. Load local image to k3s:

```bash
sudo sh -c 'docker save x3huang/chatty:local-1 | k3s ctr images import -'
sudo sh -c 'k3s ctr images ls | grep x3huang/chatty:local-1'
```

2. Rollback or delete helm release:

```bash
helm uninstall chatty -n backend
helm history chatty -n backend
```
