import React from 'react';
import './ProfileStudent.css';
import AssignmentIcon from '@material-ui/icons/Assignment';
import StarIcon from '@material-ui/icons/Star';

function ProfileStudent() {
  return (
      <div className="Profile__Student">
           
            <br/><br/>
            <div className="Profile__Content">
                <div className="Profile__Heading">
                    <AssignmentIcon className="Skill__Icon"/>
                    <p>Verified Skills</p>
                    <div className="Profile__Value">
                        MY SKILL
                    </div>
                </div>

                <div className="Profile__Heading">
                    <StarIcon className="Star__Icon"/>
                    <p>Achievements</p>
                    <div className="Profile__Value">
                        MY Achievements
                    </div>
                </div>
            </div>
      </div>
    );
}

export default ProfileStudent;