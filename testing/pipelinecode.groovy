node('master')
{
        stage('build')
        {
            bat "python --version"
            bat "git clone https://github.com/gvsmaneesha/datascience.git"
            bat "dir"
        }
}
