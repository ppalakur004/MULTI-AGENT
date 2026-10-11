import math

import pytest
from pydantic import ValidationError

from app.models import Application, ClaimsHistory, EligibilityVerification, RiskResult


@pytest.mark.parametrize(
    "field,value",
    [
        ("age", 17),
        ("age", 101),
        ("age", True),
        ("smoker", 1),
        ("coverage_tier", "GOLD"),
        ("declared_chronic_conditions", -1),
        ("declared_chronic_conditions", True),
        ("hospitalizations_last_2_years", -1),
        ("hospitalizations_last_2_years", math.inf),
        ("applicant_id", ""),
        ("applicant_id", "../x"),
        ("occupation", "x" * 121),
        ("full_name", ""),
        ("full_name", "   "),
        ("full_name", "x" * 121),
    ],
)
def test_invalid_application(field, value):
    data = dict(
        applicant_id="A",
        full_name="Test Applicant",
        age=35,
        coverage_tier="STANDARD",
        smoker=False,
        declared_chronic_conditions=0,
        hospitalizations_last_2_years=0,
    )
    data[field] = value
    with pytest.raises(ValidationError):
        Application(**data)


def test_extra_fields_and_external_bounds():
    with pytest.raises(ValidationError):
        EligibilityVerification(identity_verified=1, fraud_flag=False)
    with pytest.raises(ValidationError):
        ClaimsHistory(claims_last_2_years=1, high_cost_claims_last_2_years=2)
    with pytest.raises(ValidationError):
        Application(
            full_name="Test Applicant",
            applicant_id="A",
            age=35,
            coverage_tier="BASIC",
            smoker=False,
            declared_chronic_conditions=0,
            hospitalizations_last_2_years=0,
            injected=True,
        )
    with pytest.raises(ValidationError):
        RiskResult(score=2, band="UNKNOWN", factors=[], confidence=-1)
