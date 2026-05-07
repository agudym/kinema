from kinema.link_kinematic import LinkKinematic, generate_multicycles_robot

def test_multicycles_robot():
    links = generate_multicycles_robot(LinkKinematic, debug=True, num_cycles=10)
    # The generation function itself solves the internal kinematics initially
    # we just need to ensure it doesn't fail.
    assert len(links) > 0
