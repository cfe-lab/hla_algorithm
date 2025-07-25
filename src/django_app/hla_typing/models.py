from django.contrib.sessions.models import Session
from django.core.validators import MinValueValidator
from django.db import models

from easyhla.easyhla import HLAInterpretation


class Run(models.Model):
    """
    A discrete HLA interpretation run created by a user of the tool.

    This can represent a single HLA interpretation on a single sequence,
    or a batch job.
    """

    session = models.ForeignKey(Session, on_delete=models.CASCADE)


class Interpretation(models.Model):
    # These are job statuses:
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

    # Possible values for the locus of the sequence:
    A: str = "A"
    B: str = "B"
    C: str = "C"
    LOCI: dict[str, str] = {
        A: "A",
        B: "B",
        C: "C",
    }

    run = models.ForeignKey(Run, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    status = models.CharField(max_length=10, choices=STATUSES, default=PENDING)

    exon2 = models.CharField(max_length=500)
    intron = models.CharField(max_length=1000)
    exon3 = models.CharField(max_length=500)
    locus = models.CharField(max_length=1, choices=LOCI)


class MatchingCombinedStandard(BaseModel):
    name = models.CharField(max_length=100)
    mismatch_count = models.IntegerField(validators=[MinValueValidator(0)])
    interpretation = models.ForeignKey(Interpretation, on_delete=models.CASCADE)


class Mismatch(BaseModel):
    matching_combined_standard = models.ForeignKey(
        MatchingCombinedStandard,
        on_delete=models.CASCADE,
    )
    index = models.IntegerField(validators=[MinValueValidator(1)])
    observed_base = models.CharField(max_length=1)
    expected_base = models.CharField(max_length=1)
