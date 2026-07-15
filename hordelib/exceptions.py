class CivitAIDown(Exception):
    pass


class CivitAIResourceNotFound(Exception):
    """A CivitAI metadata request definitively failed with a client error (a not-found or unauthorized status).

    Distinct from :class:`CivitAIDown`: that signals a transient outage worth retrying, whereas this signals a
    terminal verdict (the reference does not exist, or is not accessible) that retrying the same request cannot
    change. Callers translate it into a terminal rejection rather than a retryable failure.
    """


class ModelInvalid(Exception):
    pass


class ModelEmpty(Exception):
    pass
