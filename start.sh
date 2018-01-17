# Activate the Conda environment
source activate ts-env

# Configure Git
git config --global user.email $GIT_USER_NAME
git config --global user.name $GIT_USER_MAIL

# Print information
echo ""
echo ""

echo ""
echo "To get Fish and OMF, run ..."
echo "-----------------------------------------------------------------"
echo "curl -sSL https://oh-my.fish | fish"
echo "omf install agnoster"

echo ""
echo "To get the Cookiecutter, run ..."
echo "-----------------------------------------------------------------"
echo "git clone https://github.com/dushyantkhosla/ds-template-01.git"

echo ""
echo "To start a Jupyter notebook server, run ..."
echo "-----------------------------------------------------------------"
echo "jupyter notebook --allow-root --no-browser --ip 0.0.0.0 --port 8080"


echo ""
echo "If you didn't configure git during 'docker run', run ..."
echo "-----------------------------------------------------------------"
echo "git config --global user.email 'you@example.com'"
echo "git config --global user.name 'Your Name'"
echo "git config --list"

echo ""
echo ""
