//! The adapter that puts the broker in front of a provider and a connector client.
//!
//! This is the only metered-call seam a real run uses: model calls go through the
//! broker as a connector, so there is one rate-limit accounting point and one place
//! where credentials are injected.

use std::collections::BTreeMap;

use agent_contracts::primitive::ModelRef;
use agent_contracts::types::{Name, Version};
use agent_contracts::unit::ConnectorBinding;
use serde_json::Value;

use crate::broker::Broker;
use crate::driver::MeteredCalls;
use crate::exec::Clock;
use crate::provider::{
    ConnectorClient, ModelCall, Provider, ProviderError, ProviderResponse, SecretResolver,
};

pub struct BrokerCalls<'a, C: Clock, S: SecretResolver, P: Provider, K: ConnectorClient> {
    broker: Broker<'a, C, S>,
    provider: P,
    connector: K,
}

impl<'a, C: Clock, S: SecretResolver, P: Provider, K: ConnectorClient> BrokerCalls<'a, C, S, P, K> {
    pub fn new(broker: Broker<'a, C, S>, provider: P, connector: K) -> Self {
        BrokerCalls {
            broker,
            provider,
            connector,
        }
    }

    pub fn broker(&self) -> &Broker<'a, C, S> {
        &self.broker
    }

    pub fn broker_mut(&mut self) -> &mut Broker<'a, C, S> {
        &mut self.broker
    }

    pub fn provider(&self) -> &P {
        &self.provider
    }

    /// Release the run's fair share once the run is over.
    pub fn release_run(&mut self, run_id: &str) {
        self.broker.release_run(run_id);
    }
}

impl<C: Clock, S: SecretResolver, P: Provider, K: ConnectorClient> MeteredCalls
    for BrokerCalls<'_, C, S, P, K>
{
    fn model(
        &mut self,
        run_id: &str,
        model: &ModelRef,
        call: &ModelCall,
        wall_remaining_ms: u64,
    ) -> Result<ProviderResponse, ProviderError> {
        self.broker
            .call_model(run_id, model, call, wall_remaining_ms, &mut self.provider)
            .map_err(|e| e.to_provider_error())
    }

    fn connector(
        &mut self,
        run_id: &str,
        binding: &ConnectorBinding,
        function: &str,
        input: &Value,
        wall_remaining_ms: u64,
    ) -> Result<Value, ProviderError> {
        self.broker
            .call_connector(
                run_id,
                binding,
                function,
                input,
                wall_remaining_ms,
                &mut self.connector,
            )
            .map_err(|e| e.to_provider_error())
    }

    fn used_versions(&self) -> BTreeMap<Name, Version> {
        self.broker.used_versions().clone()
    }
}
