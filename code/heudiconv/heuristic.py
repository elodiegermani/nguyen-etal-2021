import os

def create_key(template, outtype=('nii.gz',), annotation_classes=None):
    if template is None or not template:
        raise ValueError('Template must be a valid format string')
    return template, outtype, annotation_classes

def infotodict(seqinfo):
    """Heuristic evaluator for determining which runs belong where
    allowed template fields - follow python string module:
    item: index within category
    subject: participant id
    seqitem: run number during scanning
    subindex: sub index within group
    """
    func = create_key('sub-{subject}/func/sub-{subject}_task-rest_bold')
    anat = create_key('sub-{subject}/anat/sub-{subject}_T1w')

    info = {func: [], anat: []}
    
    for idx, s in enumerate(seqinfo):
        if  ('ep2d' in s.protocol_name):
            info[func].append(s.series_id)
        if ('MPRAGE' in s.protocol_name):
            info[anat].append(s.series_id)
    
    return info