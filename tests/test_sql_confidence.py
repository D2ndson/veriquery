from training.calibrate_sql_confidence import fit, metrics, wilson


def rows():
    return [{'confidence':.9 if i<4 else .3,'score':{'correct':i<4}} for i in range(10)]


def test_calibration_requires_coverage_and_accuracy():
    result=fit(rows())
    assert result['threshold']==.9
    assert fit(rows(),minimum_coverage=.5)['threshold'] is None


def test_heldout_accuracy_can_fall():
    policy=fit(rows())
    test=[{'confidence':.9,'score':{'correct':False}}]
    assert metrics(test,policy['threshold'])['accuracy']==0


def test_unscorable_and_empty_acceptance():
    r=rows()+[{'confidence':.9,'score':{'correct':None}}]
    assert metrics(r,.9)['scorable']==10
    assert metrics(r,1)['accuracy'] is None
    assert wilson(0,0)==[None,None]
    assert wilson(4,4)[0]<.85
