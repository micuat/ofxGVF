#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
    
    ofSetLogLevel(OF_LOG_VERBOSE);

    
    // CONFIGURATION of the GVF
    config.inputDimensions = 2;
    config.translate       = true;
    config.segmentation    = false;
    
    // PARAMETERS are set by default

    // CREATE the corresponding GVF
    gvf.setup(config);
	gvf.setTolerance(20.f);
    
    ofBackground(0, 0, 0);
	performingLearning = false;
    performingFollowing = false;
    
	
	templateFile = ofToDataPath("templates.txt");
	if( ofFile::doesFileExist(templateFile) ) {
		gvf.loadTemplates(templateFile);
	}
}

//--------------------------------------------------------------
void ofApp::update(){
	if (performingLearning)
		currentGesture.addObservationRaw(ofPoint(mouseX, mouseY, 0));
	
	if (performingFollowing){
		currentGesture.addObservationRaw(ofPoint(mouseX, mouseY, 0));
		gvf.infer(currentGesture.getLastRawObservation());
	}
}

//--------------------------------------------------------------
void ofApp::draw(){

    currentGesture.draw();
    
    for(int i = 0; i < gvf.getNumberOfGestureTemplates(); i++){
        
        ofxGVFGesture & gestureTemplate = gvf.getGestureTemplate(i);
        
        gestureTemplate.draw(i * 100.0f, ofGetHeight() - 100.0f, 100.0f, 100.0f);
        
    }
    
    gvf.drawParticles(currentGesture);
    
	
    ofSetColor(255, 255, 255);
    
    ostringstream os;
    os << "GESTURE VARIATION FOLLOWER 2D Example " << endl;
    os << "FPS: " << ofGetFrameRate() << endl;
    os << "GVFState: " << gvf.getStateAsString() << " ('l': learning, 'f': following, 'c': Clear)" << endl;
    os << "Gesture Recognized: " << gvf.getMostProbableGestureIndex()+1 << endl;
    
    
    float phase = 0.0f;
    float speed = 0.0f;
    float size  = 0.0f;
    float angle = 0.0f;
    

    // if performing gesture in following mode, display estimated variations
    if (performingFollowing)
    {
        // get outcomes: estimations of how the gesture is performed
        outcomes = gvf.getOutcomes();
        
		if (outcomes.most_probable >= 0){
			phase = outcomes.estimations[outcomes.most_probable].phase;
			speed = outcomes.estimations[outcomes.most_probable].speed;
			size  = outcomes.estimations[outcomes.most_probable].scale[0];
			angle = outcomes.estimations[outcomes.most_probable].rotation[0];
		}
    }
  
    os << "Cursor: " << phase << " | Speed: " << speed << " | Size: " << size << " | Angle: " << angle << endl;
    
    ofDrawBitmapString(os.str(), 20, 20);
    
}


//--------------------------------------------------------------
void ofApp::keyPressed(int key){
    switch(key){
        case 'L':
        case 'l':
            gvf.setState(ofxGVF::STATE_LEARNING);
            break;
        case 'f':
            gvf.setState(ofxGVF::STATE_FOLLOWING);
            break;
        case 'c':
            gvf.setState(ofxGVF::STATE_CLEAR);
            break;
        case 's':
            gvf.saveTemplates(templateFile);
            break;
        case 'g':
            currentGesture.setType(ofxGVFGesture::GEOMETRIC);
            break;
        case 't':
            currentGesture.setType(ofxGVFGesture::TEMPORAL);
            break;
        default:
            break;
    }
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){
    
}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){
	
	performingFollowing = false;
	performingLearning = false;
	
	// initialize gesture
	if( gvf.getState() == ofxGVF::STATE_LEARNING ||
	    gvf.getState() == ofxGVF::STATE_FOLLOWING ) {
		currentGesture.clear();
		currentGesture.setAutoAdjustRanges(false);
		currentGesture.setMin(0.0f, 0.0f);
		currentGesture.setMax(ofGetWidth(), ofGetHeight());
	}
	
    switch(gvf.getState()){
        case ofxGVF::STATE_LEARNING:
		{
            performingLearning = true;
            break;
		}
        case ofxGVF::STATE_FOLLOWING:
		{
            gvf.spreadParticles();
            performingFollowing = true;
            break;
		}
    }
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){
    
    performingLearning = false;
    performingFollowing = false;
	
    switch(gvf.getState()){
        case ofxGVF::STATE_LEARNING:
        {
            gvf.addGestureTemplate(currentGesture);
            break;
        }
        case ofxGVF::STATE_FOLLOWING:
        {
            gvf.spreadParticles();
            break;
        }
    }
	
	currentGesture.clear();
}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}
