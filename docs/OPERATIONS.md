# Operational Best Practices — MLOps / LLMOps

This document outlines production-grade recommendations for running the
AI-Powered Last Mile Delivery Automation system on AWS ECS Fargate.

---

## 1. Secrets Management

| What | Where | Why |
|------|-------|-----|
| `OPENAI_API_KEY`, `OPENAI_API_BASE` | AWS Secrets Manager | Never bake API keys into Docker images or config files |
| `LANGCHAIN_API_KEY` | AWS Secrets Manager | LangSmith tracing key |
| `CHROMA_API_KEY`, `CHROMA_TENANT`, `CHROMA_DATABASE` | AWS Secrets Manager | ChromaDB Cloud credentials |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | GitHub Secrets | CI/CD pipeline authentication only |

### How it works

1. Store all application secrets as a single JSON secret in AWS Secrets Manager
   (e.g., `last-mile-delivery/api-keys`).
2. In the ECS Task Definition, reference individual keys via the `secrets` block:

   ```json
   "secrets": [
     {
       "name": "OPENAI_API_KEY",
       "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789:secret:last-mile-delivery/api-keys:OPENAI_API_KEY::"
     },
     {
       "name": "CHROMA_API_KEY",
       "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789:secret:last-mile-delivery/api-keys:CHROMA_API_KEY::"
     }
   ]
   ```

3. ECS injects these as environment variables at container startup — no code
   changes needed, since `ModelLoader` already reads from `os.getenv()`.

### IAM Requirements

The ECS **Task Execution Role** needs:
```json
{
  "Effect": "Allow",
  "Action": ["secretsmanager:GetSecretValue"],
  "Resource": "arn:aws:secretsmanager:us-east-1:*:secret:last-mile-delivery/*"
}
```

---

## 2. Rollback Strategy

### Automatic Rollback via ECS Circuit Breaker

Enable the deployment circuit breaker in the ECS service configuration:

```json
{
  "deploymentConfiguration": {
    "deploymentCircuitBreaker": {
      "enable": true,
      "rollback": true
    },
    "maximumPercent": 200,
    "minimumHealthyPercent": 100
  }
}
```

**How it works:**
- ECS monitors new task health during deployment.
- If new tasks repeatedly fail to reach a healthy state, the circuit breaker
  triggers and automatically rolls back to the last stable task definition.
- The CI/CD pipeline's `wait-for-service-stability` step will report the
  failure, giving clear signal in GitHub Actions.

### Manual Rollback

Since every image is tagged with its commit SHA, you can redeploy any previous
version:

```bash
# Find the last known-good commit SHA
git log --oneline -10

# Update the ECS service to the previous image
aws ecs update-service \
  --cluster last-mile-delivery-cluster \
  --service last-mile-delivery-service \
  --force-new-deployment \
  --task-definition <previous-task-def-arn>
```

---

## 3. Infrastructure as Code (IaC)

We recommend managing all AWS resources with **Terraform** or **AWS CDK**.

### Resources to codify

| Resource | Purpose |
|----------|---------|
| `aws_ecr_repository` | Docker image registry |
| `aws_ecs_cluster` | Fargate cluster |
| `aws_ecs_service` | Service definition with circuit breaker, ALB target group |
| `aws_ecs_task_definition` | Container config, secrets refs, log driver, health check |
| `aws_lb` + `aws_lb_target_group` | Application Load Balancer for HTTPS termination |
| `aws_cloudwatch_log_group` | Centralized log destination |
| `aws_secretsmanager_secret` | API keys and credentials |
| `aws_iam_role` | Task execution role (pulls images + secrets) and task role |

### Example Terraform snippet (ECS task definition)

```hcl
resource "aws_ecs_task_definition" "app" {
  family                   = "last-mile-delivery-task"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 1024
  memory                   = 2048
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name      = "last-mile-delivery-app"
    image     = "${aws_ecr_repository.app.repository_url}:latest"
    essential = true

    portMappings = [{ containerPort = 8080, protocol = "tcp" }]

    environment = [
      { name = "ENV",         value = "production" },
      { name = "LOG_FORMAT",  value = "json" },
      { name = "PROJECT_ROOT", value = "/app" },
    ]

    secrets = [
      { name = "OPENAI_API_KEY",  valueFrom = "${aws_secretsmanager_secret.api_keys.arn}:OPENAI_API_KEY::" },
      { name = "OPENAI_API_BASE", valueFrom = "${aws_secretsmanager_secret.api_keys.arn}:OPENAI_API_BASE::" },
      { name = "CHROMA_API_KEY",  valueFrom = "${aws_secretsmanager_secret.api_keys.arn}:CHROMA_API_KEY::" },
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.app.name
        "awslogs-region"        = "us-east-1"
        "awslogs-stream-prefix" = "ecs"
      }
    }

    healthCheck = {
      command     = ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8080/health')\""]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 10
    }
  }])
}
```

---

## 4. Logging & Monitoring

### CloudWatch Logs (structured JSON)

The application's `LOG_FORMAT=json` env var (set in the Dockerfile) activates
the JSON formatter in `logging_config.py`. Combined with the ECS `awslogs`
driver, all container output flows to CloudWatch Logs as structured JSON,
enabling:
- **CloudWatch Logs Insights** queries (e.g., filter by agent name, error level)
- **Metric filters** to create alarms from log patterns

### LangSmith Tracing (LLM Observability)

Already wired in the codebase. Set these env vars in the ECS task definition:
- `LANGCHAIN_API_KEY` — your LangSmith API key
- `LANGCHAIN_PROJECT` — `"Last-Mile Logistics"`
- `LANGCHAIN_TRACING_V2` — `"true"`

LangSmith provides:
- Per-agent trace visualization (resolution, communication, critic nodes)
- Token usage and latency tracking per LLM call
- Evaluation dataset management and regression testing

### Recommended CloudWatch Alarms

| Alarm | Metric | Threshold |
|-------|--------|-----------|
| High CPU | `ECSService` CPUUtilization | > 80% for 5 min |
| High Memory | `ECSService` MemoryUtilization | > 85% for 5 min |
| 5xx Spike | ALB `HTTPCode_Target_5XX_Count` | > 10 in 5 min |
| Task Restarts | `ECSService` RunningTaskCount drops | < desired count |
| Health Check Fail | ALB `UnHealthyHostCount` | > 0 for 3 min |

### Log Insights Query Examples

```sql
-- Find all errors in the last hour
fields @timestamp, @message
| filter level = "ERROR"
| sort @timestamp desc
| limit 50

-- Agent latency breakdown
fields @timestamp, agent_name, latency_sec
| filter agent_name IN ["resolution", "communication", "critic"]
| stats avg(latency_sec) as avg_latency by agent_name

-- Token cost tracking
fields @timestamp, prompt_tokens, completion_tokens, estimated_cost
| stats sum(estimated_cost) as total_cost by bin(1h)
```

---

## 5. GitHub Secrets Checklist

Configure these in your GitHub repository under **Settings → Secrets and variables → Actions**:

| Secret Name | Description |
|-------------|-------------|
| `AWS_ACCESS_KEY_ID` | IAM user with ECR push + ECS deploy permissions |
| `AWS_SECRET_ACCESS_KEY` | Corresponding secret key |

> **Tip:** For better security, consider using GitHub OIDC with
> `aws-actions/configure-aws-credentials` to avoid long-lived IAM credentials.
> This requires an IAM Identity Provider trust policy instead of access keys.
