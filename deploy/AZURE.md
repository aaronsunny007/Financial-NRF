# Deploying the results explorer on Azure Container Apps (free tier)

The demo is a Gradio viewer over the saved files in `results/`. It needs no model
and no GPU. The container image is built by `.github/workflows/deploy-demo.yml`
and stored in GitHub Container Registry (free). Azure Container Apps runs it on the
Consumption plan, which has a monthly free grant and scales to zero when idle.

Do not use Azure Container Registry: it is a paid service.

> Free-tier terms change. Before you start, check the current "Azure Container Apps
> pricing" page and your Azure free-account terms. An account needs a credit card
> for identity verification.

## 0. Before anything else

1. Merge the research branch into `master`, so the image contains the current
   "Paper findings" tab.
2. Merge the branch that adds this file. The workflow only runs from `master`.
3. After the first successful run of **deploy-demo**, open
   GitHub -> your profile -> Packages -> `financial-nrf-demo` -> Package settings ->
   **Change visibility -> Public**. Azure then pulls the image with no credentials.

## 1. Create the Azure account and a budget alert

1. Sign up at azure.microsoft.com/free.
2. Cost Management + Billing -> Budgets -> create a budget of 5 USD/month with an
   email alert at 50% and 100%. Do this first.

## 2. One-time resource creation (Azure CLI or Cloud Shell)

```bash
az login
az group create --name nrf-demo-rg --location northeurope
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
```

```bash
az containerapp env create --name nrf-demo-env --resource-group nrf-demo-rg --location northeurope
```

```bash
az containerapp create \
  --name nrf-demo \
  --resource-group nrf-demo-rg \
  --environment nrf-demo-env \
  --image ghcr.io/aaronsunny007/financial-nrf-demo:latest \
  --target-port 7860 \
  --ingress external \
  --min-replicas 0 --max-replicas 1 \
  --cpu 0.5 --memory 1.0Gi
```

The last command prints the app's public URL. `--min-replicas 0` lets it scale to
zero, so idle time costs nothing. The first visit after idle takes a few seconds.

## 3. Let GitHub deploy new versions (OIDC, no stored passwords)

```bash
az ad app create --display-name nrf-demo-github
```

Then create a service principal for the app, grant it `Contributor` on the
`nrf-demo-rg` resource group only, and add a federated credential for
`repo:aaronsunny007/Financial-NRF:ref:refs/heads/master`. Microsoft's guide
"Use GitHub Actions to connect to Azure" has the exact commands.

In the GitHub repo settings add:

| Kind | Name | Value |
|---|---|---|
| Secret | `AZURE_CLIENT_ID` | the app's client id |
| Secret | `AZURE_TENANT_ID` | your tenant id |
| Secret | `AZURE_SUBSCRIPTION_ID` | your subscription id |
| Variable | `AZURE_APP_NAME` | `nrf-demo` |
| Variable | `AZURE_RESOURCE_GROUP` | `nrf-demo-rg` |
| Variable | `AZURE_DEPLOY` | `true` |

From then on, every merge to `master` that touches `deploy/`, `results/` or the
Dockerfile rebuilds the image and rolls the app.

## 4. Tear down when you are done

```bash
az group delete --name nrf-demo-rg --yes
```

This removes every resource in the group, so nothing keeps billing.

## Run it locally first

```bash
docker build -t nrf-demo .
docker run -p 7860:7860 nrf-demo
```
