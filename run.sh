export no_proxy="localhost, 127.0.0.1"
export CLLM_SERVICES_PORT=10004
export TOG_SERVICE_PORT=10005
export GRADIO_TEMP_DIR="~/.tmp"
export OPENAI_API_KEY="sk-xxx"
export OPENAI_BASE_URL="xxx"
export WEATHER_API_KEY="xxx"
export TASK_DECOMPOSITION_CKPT="./model_zoo/task_decomposition"
export CLIENT_ROOT="./client_resources"
export SERVER_ROOT="./server_resources"

echo "Launch all tool services..."
# step 1
python -m cllm.services.launch --port $CLLM_SERVICES_PORT --host 0.0.0.0 &

echo "Launch ToG service..."
# step 2
python -m cllm.services.tog.launch --port $TOG_SERVICE_PORT --host 0.0.0.0 &

echo "Launch gradio demo..."
# step 3
python -m cllm.app.gradio --controller "cllm.agents.tog.Controller" --server-port 10003
# python -m cllm.app.gradio --controller "cllm.agents.tog.Controller" --server-port 10003 --https
