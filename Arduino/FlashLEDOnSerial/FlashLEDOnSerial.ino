/*
  FlashLEDOnSerial
  Turns on an LED for photometric stereo data acquisition using 3D plant 
  phenotyping system - PS-Plant. The is procedure is as follows:
    * MKRZero receives LED number (0-7) from the controller;
    * Switches off all lights;
    * Switches on corresponding LED;
    * Returns the received LED number.
  
  If the last command was received over 'SIGNAL_CUTOFF' ms ago and the lights
  are on, the lights are switched off. 
  
  If LED number '9' is received, all lights are switched off - typically after
  the image was captured.

  modified 21 Nov 2018
  by Gytis Bernotas
*/
int N_PINS=8;
int LEDPins[]={0,18,1,19, 4,2,5,3}; // for VIS (4) + NIR (4) = VISNIR (8)
int SIGNAL_CUTOFF=500; // All lights are powered off the arduino receives no instruction for 0.5s
int PWM_VIS=255; // The PWM level for the visible lights, 255 is full duty cycle, 122 would be half, 0 is off
int PWM_NIR=255; // The PWM level for the NIR lights, 255 is full duty cycle, 122 would be half, 0 is off
bool LIGHTS_ON=false; // Flag to prevent unnecessary switching off the lights every loop when they're already off

void setup() 
{
  Serial.begin(9600);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB port only
  }
}

void switchOffLights() 
{
  for (int i=0; i<N_PINS; i++)
     analogWrite(LEDPins[i], 0);
  LIGHTS_ON=false;
}

void switchOnLight(int light) 
{
  // Which PWM brightness to use based on the index of the light (0-3 are visible, 4-7 are NIR)
  int pwmLevel = light < 4 ? PWM_VIS : PWM_NIR;
  analogWrite(LEDPins[light], pwmLevel);
  LIGHTS_ON=true;

}
void handleSerialOptions(char serChar) {
  switch(serChar){
    case 48: // '0'
      switchOffLights();
      switchOnLight(0);
      break;
    case 49: // '1'
      switchOffLights();
      switchOnLight(1);
      break;
    case 50: // '2'
      switchOffLights();
      switchOnLight(2);
      break;
    case 51: // '3'
      switchOffLights();
      switchOnLight(3);
      break;
    case 52: // '4'
      switchOffLights();
      switchOnLight(4);
      break;
    case 53: // '5'
      switchOffLights();
      switchOnLight(5);
      break;
    case 54: // '6'
      switchOffLights();
      switchOnLight(6);
      break;
    case 55: // '7'
      switchOffLights();
      switchOnLight(7);
      break;

    case 57: // '9'
      switchOffLights();
      break;
  }
}

void loop() 
{
  static long lastSignalReceived=millis();
  if (Serial.available()) 
  {
    int serChar=Serial.read();
    handleSerialOptions(serChar);
    Serial.write(serChar); // Just send back what we received as an ACK
    lastSignalReceived=millis();
  }
  
  if ((millis() - lastSignalReceived) > SIGNAL_CUTOFF)
  {
    if (LIGHTS_ON) 
    {
       switchOffLights();
    }
  }
}



