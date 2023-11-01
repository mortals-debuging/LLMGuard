import grpc
import service_pb2
import service_pb2_grpc

class MyService(service_pb2_grpc.MyServiceServicer):
    def MyFunction(self, request, context):
        return service_pb2.MyResponse(output="Hello, " + request.input)

server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
service_pb2_grpc.add_MyServiceServicer_to_server(MyService(), server)
server.add_insecure_port('[::]:50051')
server.start()
server.wait_for_termination()