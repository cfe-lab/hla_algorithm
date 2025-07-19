from django.contrib.auth.models import User
from django.db import models


class Run(models.Model):
    """
    A discrete HLA interpretation run created by a user of the tool.

    This can represent a single HLA interpretation on a single sequence,
    or a batch job.
    """

    user = models.ForeignKey(User, on_delete=models.CASCADE)


class Interpretation(models.Model):
    RUNNING: str = "RUNNING"
    CANCELLED: str = "CANCELLED"
    PENDING: str = "PENDING"
    COMPLETE: str = "COMPLETE"
    FAILED: str = "FAILED"
    STATUSES: dict[str, str] = {
        RUNNING: "running",
        CANCELLED: "cancelled",
        PENDING: "pending",
        COMPLETE: "complete",
        FAILED: "failed",
    }

    run = models.ForeignKey(Run, on_delete=models.CASCADE)
    status = models.CharField(choices=STATUSES, default=PENDING)


class HLASequence(models.Model):
    interpretation = models.ForeignKey(Interpretation, on_delete=models.CASCADE)
