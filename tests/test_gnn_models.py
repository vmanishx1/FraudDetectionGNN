import pytest
import torch
from ml.models.gnn_models import GCN, GraphSAGE, GAT, ModelFactory


class TestGNNModels:
    def setup_method(self):
        self.in_channels = 16
        self.num_nodes = 10
        self.x = torch.randn(self.num_nodes, self.in_channels)
        self.edge_index = torch.randint(0, self.num_nodes, (2, 30))

    def test_gcn_forward(self):
        model = GCN(in_channels=self.in_channels)
        out = model(self.x, self.edge_index)
        assert out.shape == (self.num_nodes, 2)

    def test_graphsage_forward(self):
        model = GraphSAGE(in_channels=self.in_channels)
        out = model(self.x, self.edge_index)
        assert out.shape == (self.num_nodes, 2)

    def test_gat_forward(self):
        model = GAT(in_channels=self.in_channels)
        out = model(self.x, self.edge_index)
        assert out.shape == (self.num_nodes, 2)

    def test_model_factory(self):
        for model_type in ["gcn", "graphsage", "gat"]:
            model = ModelFactory.create_model(model_type, in_channels=self.in_channels)
            assert model is not None
            out = model(self.x, self.edge_index)
            assert out.shape[-1] == 2

    def test_model_factory_invalid(self):
        with pytest.raises(ValueError):
            ModelFactory.create_model("invalid_model", in_channels=self.in_channels)

    def test_different_hidden_dim(self):
        model = GCN(in_channels=self.in_channels, hidden_dim=64)
        out = model(self.x, self.edge_index)
        assert out.shape == (self.num_nodes, 2)
