"""Single-request risk acknowledgement; never modifies saved training config."""
FLAG = '--allow_unverified_nf4_compile'
RISK_CODE = 'mixed_nf4_compile'


def without_risk_flag(extra_args):
    return [arg for arg in (extra_args or [])
            if not isinstance(arg, str) or not arg.startswith(FLAG)]


def remaining_preflight(preflight, *, accepted=False):
    errors = [item for item in (preflight.get('errors') or [])
              if not (accepted and item.get('overridable') is True
                      and item.get('risk_code') == RISK_CODE)]
    return {**preflight, 'errors': errors, 'ok': not errors and (bool(preflight.get('ok')) or bool(preflight.get('errors'))),
            'summary': {**preflight.get('summary', {}), 'errors': len(errors)}}


def acknowledge_request_risks(data, preflight, extra_args):
    accepted = data.get('ignore_risks') is True and data.get('confirmed') is True
    args = without_risk_flag(extra_args)
    if accepted and any(item.get('overridable') is True and item.get('risk_code') == RISK_CODE
                        for item in preflight.get('errors') or []):
        args.append(FLAG)
    return remaining_preflight(preflight, accepted=accepted), args


def pending_preflight(preflight, extra_args):
    return remaining_preflight(preflight, accepted=FLAG in (extra_args or []))
