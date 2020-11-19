import logging
import time
from jaeger_client import Config
import atexit

tracer = None
root_span = None


def _init(service):
    logging.getLogger('').handlers = []
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)

    config = Config(
        config={ # usually read from some yaml config
            'sampler': {
                'type': 'const',
                'param': 1,
            },
            'reporter_batch_size': 1,
            'local_agent': {
                'reporting_host': '10.195.24.21',
                # 'reporting_port': '16686',
            },
            'logging': True,
        },
        service_name='deepspeed',
        validate=True,
    )
    # this call also sets opentracing.tracer
    return config.initialize_tracer()


def init(service="deepspeed"):
    global tracer
    global root_span
    print("initialized tracer...")
    tracer = _init(service)
    root_span = tracer.start_span('root')

    def close():
        global root_span
        global tracer
        # print("closing tracer...")
        # print(tracer)
        time.sleep(1)
        root_span.finish()
        time.sleep(2)
        tracer.close()

    atexit.register(close)
    return tracer


def get_tracer():
    return tracer


if __name__ == "__main__":
    log_level = logging.DEBUG
    logging.getLogger('').handlers = []
    logging.basicConfig(format='%(asctime)s %(message)s', level=log_level)

    # this call also sets opentracing.tracer
    tracer = _init('first-service')

    with tracer.start_span('TestSpan') as span:
        span.log_kv({'event': 'test message', 'life': 42})

        with tracer.start_span('ChildSpan', child_of=span) as child_span:
            child_span.log_kv({'event': 'down below'})

    time.sleep(
        10
    )  # yield to IOLoop to flush the spans - https://github.com/jaegertracing/jaeger-client-python/issues/50
    tracer.close()  # flush any buffered spans
