	node('master')
	{
	        stage('check version')
	        {
	            bat "python --version"
	                  }
	stage('checkout')
	        {
	                        bat "git clone https://github.com/gvsmaneesha/datascience.git"
	         
	        }
	
	stage('check version')
	        {
		
		bat "cd datascience/testing"   
		bat "python sample.py"    
	
	        }
	stage('successful')
	        {
		
		Print("Successfully executed")
	        }
	
	
	}
	
