"""
honeypots.py — Programmatic honeypot detection for Redrob Hackathon.

Checks every candidate profile for synthetic anomalies that signal
a honeypot or otherwise impossible/fraudulent profile.
Returns True if the candidate is flagged as a honeypot (score = 0).
"""
from datetime import date, datetime
from typing import Dict, Any

REFERENCE_DATE = date(2026, 6, 5)


def _parse_date(s) -> date:
    if s is None:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def _months_between(d1: date, d2: date) -> int:
    """Return approximate months between two dates."""
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)


def is_honeypot(candidate: Dict[str, Any]) -> bool:
    """
    Return True if the candidate is a synthetic honeypot.
    Any single failed check flags the entire profile as a honeypot.
    """
    reasons = []

    # ------------------------------------------------------------------ #
    # CHECK 1: Career timeline math mismatch                              #
    # For each job, the declared duration_months must be consistent with  #
    # start_date / end_date. Flag if off by > 6 months AND > 20%.        #
    # ------------------------------------------------------------------ #
    for job in candidate.get("career_history", []):
        start = _parse_date(job.get("start_date"))
        end_raw = job.get("end_date")
        end = _parse_date(end_raw) if end_raw else REFERENCE_DATE
        declared = job.get("duration_months", 0)

        if start is None or end is None or end < start:
            continue  # skip unparsable dates

        actual = _months_between(start, end)
        diff = abs(declared - actual)
        pct = diff / max(actual, 1)
        if diff > 6 and pct > 0.20:
            reasons.append(
                f"career_duration_mismatch: job start={job.get('start_date')} "
                f"end={end_raw} declared={declared} actual≈{actual}"
            )

    # ------------------------------------------------------------------ #
    # CHECK 2: Expert/Advanced skill with near-zero usage duration        #
    # If 3 or more high-proficiency skills have duration_months ≤ 3      #
    # the profile is suspicious.                                          #
    # ------------------------------------------------------------------ #
    suspicious_skills = 0
    for skill in candidate.get("skills", []):
        prof = skill.get("proficiency", "")
        dur = skill.get("duration_months", 0)
        if prof in ("expert", "advanced") and dur <= 3:
            suspicious_skills += 1
    if suspicious_skills >= 3:
        reasons.append(
            f"skill_duration_anomaly: {suspicious_skills} expert/advanced "
            f"skills with ≤3 months duration"
        )

    # ------------------------------------------------------------------ #
    # CHECK 3: Education date inconsistency                               #
    # Graduation year must be ≥ enrolment year.                          #
    # ------------------------------------------------------------------ #
    for edu in candidate.get("education", []):
        sy = edu.get("start_year")
        ey = edu.get("end_year")
        if sy and ey and sy > ey:
            reasons.append(f"edu_date_reversed: start={sy} > end={ey}")

    # ------------------------------------------------------------------ #
    # CHECK 4: Impossible platform activity signals                       #
    # ------------------------------------------------------------------ #
    sig = candidate.get("redrob_signals", {})

    signup = _parse_date(sig.get("signup_date"))
    last_active = _parse_date(sig.get("last_active_date"))
    if signup and last_active and last_active < signup:
        reasons.append(
            f"activity_before_signup: last_active={sig.get('last_active_date')} "
            f"< signup={sig.get('signup_date')}"
        )

    if signup and signup > REFERENCE_DATE:
        reasons.append(f"future_signup_date: {sig.get('signup_date')}")

    sal = sig.get("expected_salary_range_inr_lpa", {})
    sal_min = sal.get("min", 0)
    sal_max = sal.get("max", 0)
    if sal_min > 0 and sal_max > 0 and sal_min > sal_max:
        reasons.append(
            f"salary_range_inverted: min={sal_min} > max={sal_max}"
        )

    # ------------------------------------------------------------------ #
    # CHECK 5: Negative or out-of-range numeric fields                    #
    # ------------------------------------------------------------------ #
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)
    if yoe < 0:
        reasons.append(f"negative_yoe: {yoe}")

    notice = sig.get("notice_period_days", 0)
    if notice < 0 or notice > 180:
        reasons.append(f"invalid_notice_period: {notice}")

    # ------------------------------------------------------------------ #
    # CHECK 6: Any single skill duration exceeds total career by > 1 yr  #
    # ------------------------------------------------------------------ #
    total_exp_months = int(yoe * 12)
    for skill in candidate.get("skills", []):
        skill_dur = skill.get("duration_months", 0)
        if total_exp_months > 0 and skill_dur > total_exp_months + 12:
            reasons.append(
                f"skill_duration_exceeds_career: skill={skill.get('name')} "
                f"duration={skill_dur} vs career={total_exp_months}m"
            )
            break  # one is enough to flag

    return len(reasons) > 0, reasons
